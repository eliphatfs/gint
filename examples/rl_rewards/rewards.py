"""20 heterogeneous reward kernels modelled on ManiSkill / dm_control tasks.

Each task computes a per-scene scalar reward from a 16-float state vector.
We provide two implementations of every reward:

* a ``@bytecode`` gint program that runs all 128 scenes in a single warp
  (32 lanes x REG_WIDTH=4 width-lanes = 128 elements per dispatch);
* a numerically-equivalent ``torch`` function that mirrors the math
  exactly so the two paths can be compared bit-for-bit-ish.

The shared 16-feature ABI is documented in ``FEATURE_LAYOUT`` below; each
task uses a distinct subset and combines them with different math
(distance + tanh, exp tracking, multiplicative tolerances, masked
overrides, finger-spread matches, etc.) so the resulting bytecodes are
genuinely heterogeneous, not just renamed copies.

Used by ``examples.rl_rewards.__main__`` to benchmark gint
``execute_indirect`` vs eager-serial torch dispatch on heterogeneous
RL-style workloads.
"""
import torch
from gint import TensorInterface, bytecode
from gint.host.frontend import (
    make_block_1d, fldg_1d, fstg_1d, fadd, fmul, fsub, fneg,
    faddimm, fmulimm, fpush, frcp, dup, fselect,
    fsqrt, fexp, fcos, flt, fgt, halt,
)


NUM_TASKS = 20
NUM_SCENES = 128            # one warp processes one task's full batch
NUM_FEATURES = 16           # padded ABI shared by every task

# Shared 16-feature state vector ABI.  Reward kernels read whichever subset
# they need and ignore the rest.  Keeping the ABI uniform is what lets us
# pack all tasks into a single contiguous (NUM_TASKS, NUM_FEATURES,
# NUM_SCENES) tensor and dispatch them through a single indirect launch.
FEATURE_LAYOUT = {
    'tcp_x':    0, 'tcp_y':    1, 'tcp_z':    2,
    'obj_x':    3, 'obj_y':    4, 'obj_z':    5,
    'goal_x':   6, 'goal_y':   7, 'goal_z':   8,
    'scalar_a': 9,  'scalar_b': 10, 'scalar_c': 11, 'scalar_d': 12,
    'scalar_e': 13, 'scalar_f': 14, 'scalar_g': 15,
}


# -------------- gint bytecode helpers --------------
#
# Layout: the per-task state slice has shape [16, 128] (feature, scene),
# contiguous, so feature k starts at scalar offset k * 128.  We expose it
# via a 1D block of length 16 * 128 = 2048 with stride 1; addressing
# state[k, scene] is just an offset of k*128 into that block, plus the
# usual lane/width-lane encoding the kernel applies on top.  Each warp
# processes all 128 scenes for one task in one pass (32 lanes * REGW=4).

def _F(sblock, k):
    """Load 128 scenes for feature k onto the stack (1 stack value)."""
    fldg_1d(k * NUM_SCENES, sblock)


def _sq_diff(sblock, a, b):
    """Push (state[a]-state[b])**2."""
    _F(sblock, a); _F(sblock, b); fsub(); dup(); fmul()


def _sq_dist3(sblock, a_base, b_base):
    """Push squared L2 distance over a 3-vector (a..a+2 vs b..b+2)."""
    _sq_diff(sblock, a_base + 0, b_base + 0)
    _sq_diff(sblock, a_base + 1, b_base + 1); fadd()
    _sq_diff(sblock, a_base + 2, b_base + 2); fadd()


def _sq_dist2(sblock, a_base, b_base):
    """Push squared L2 distance over a 2-vector (a..a+1 vs b..b+1)."""
    _sq_diff(sblock, a_base + 0, b_base + 0)
    _sq_diff(sblock, a_base + 1, b_base + 1); fadd()


def _inv_tanh_macro(scale: float):
    """Replace top-of-stack x with 1 - tanh(scale*x).

    Computed as 2 / (exp(2*scale*x) + 1) to keep the gint and torch paths
    bit-identical (we use the same identity in the torch reference).
    Stack effect: 0.
    """
    fmulimm(2.0 * scale)
    fexp()
    faddimm(1.0)
    frcp()
    fmulimm(2.0)


def _tanh_macro(scale: float):
    """Replace top-of-stack x with tanh(scale*x). Computed as
    1 - 2/(exp(2*scale*x)+1).  Stack effect: 0."""
    _inv_tanh_macro(scale)
    fneg()
    faddimm(1.0)


def _reach3(sblock, a_base, b_base, scale=5.0):
    """Push 1 - tanh(scale * sqrt(sq_dist3(a,b)))."""
    _sq_dist3(sblock, a_base, b_base); fsqrt(); _inv_tanh_macro(scale)


def _reach2(sblock, a_base, b_base, scale=5.0):
    """Push 1 - tanh(scale * sqrt(sq_dist2(a,b)))."""
    _sq_dist2(sblock, a_base, b_base); fsqrt(); _inv_tanh_macro(scale)


# -------------- torch reference helpers --------------
#
# We deliberately mirror gint's ``1 - tanh(x) = 2/(exp(2x)+1)`` identity
# and our explicit sqrt-of-squared-sum, so the two paths produce the same
# float32 bits.  Using ``torch.tanh``/``torch.linalg.norm`` directly would
# match in IEEE limit but drift by a few ulps in practice.

def _t_inv_tanh(x, scale=5.0):
    return 2.0 / (torch.exp(2.0 * scale * x) + 1.0)


def _t_tanh(x, scale=5.0):
    return 1.0 - _t_inv_tanh(x, scale)


def _t_sq_dist3(s, a, b):
    return ((s[a]-s[b])**2 + (s[a+1]-s[b+1])**2 + (s[a+2]-s[b+2])**2)


def _t_sq_dist2(s, a, b):
    return ((s[a]-s[b])**2 + (s[a+1]-s[b+1])**2)


def _t_reach3(s, a, b, scale=5.0):
    return _t_inv_tanh(torch.sqrt(_t_sq_dist3(s, a, b)), scale)


def _t_reach2(s, a, b, scale=5.0):
    return _t_inv_tanh(torch.sqrt(_t_sq_dist2(s, a, b)), scale)


# Indices used across the rewards.  TCP/OBJ/GOAL keep their canonical
# slots in every task; remaining features carry task-specific scalars.
T = 0   # tcp_xyz starts here
O = 3   # obj_xyz
G = 6   # goal_xyz


# ============== 20 REWARD DEFINITIONS ==============
#
# Each pair (gint @bytecode kernel, torch reference) follows the same
# pattern: read features by name, build the reward via the helpers
# above, write the per-scene result.  The math is intentionally
# different from one task to the next to exercise distinct opcode
# sequences inside ``execute_indirect``.

# ---- 01 PickCube ----------------------------------------------------
@bytecode
def gint_01_pickcube(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O)                       # acc
    _F(sblock, 10); fadd()                      # +grasp
    _F(sblock, 10); _reach3(sblock, O, G); fmul(); fadd()  # +g*place
    _F(sblock, 11); _F(sblock, 9); _inv_tanh_macro(5.0); fmul(); fadd()  # +placed*still
    fpush(5.0); _F(sblock, 12); fselect()       # success override
    fstg_1d(0, rblock); halt()


def torch_01_pickcube(s):
    acc = _t_reach3(s, T, O)
    g = s[10]; p = s[11]; succ = s[12]
    acc = acc + g
    acc = acc + g * _t_reach3(s, O, G)
    acc = acc + p * _t_inv_tanh(s[9], 5.0)
    return torch.where(succ > 0, torch.full_like(acc, 5.0), acc)


# ---- 02 StackCube ---------------------------------------------------
@bytecode
def gint_02_stackcube(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O); fmulimm(2.0)         # 2*reach(tcp,A)
    _F(sblock, 10); _reach3(sblock, O, G); fmul(); fmulimm(2.0); fadd()  # +2g*reach(A,B)
    fpush(8.0); _F(sblock, 12); fselect()       # success_on_top -> 8
    fstg_1d(0, rblock); halt()


def torch_02_stackcube(s):
    acc = 2.0 * _t_reach3(s, T, O) + 2.0 * s[10] * _t_reach3(s, O, G)
    return torch.where(s[12] > 0, torch.full_like(acc, 8.0), acc)


# ---- 03 PushCube ----------------------------------------------------
# (1 - tanh(5d3(tcp,obj))) + reach_mask * (1 - tanh(5d2(obj.xy,goal.xy)))
@bytecode
def gint_03_pushcube(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O)                       # reach
    # mask: d2(obj.xy, goal.xy) < 0.05
    _sq_dist2(sblock, O, G); fsqrt(); fpush(0.05); fgt()  # mask = 0.05 > d
    _reach2(sblock, O, G); fmul(); fadd()
    fstg_1d(0, rblock); halt()


def torch_03_pushcube(s):
    reach = _t_reach3(s, T, O)
    d2 = torch.sqrt(_t_sq_dist2(s, O, G))
    mask = (d2 < 0.05).to(torch.float32)
    return reach + mask * _t_reach2(s, O, G)


# ---- 04 PullCube ----------------------------------------------------
# variant: r = reach3(tcp, obj+epsilon) * 0.7 + g * reach2(obj.xy, goal.xy)
@bytecode
def gint_04_pullcube(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O); fmulimm(0.7)
    _F(sblock, 10); _reach2(sblock, O, G); fmul(); fadd()
    fstg_1d(0, rblock); halt()


def torch_04_pullcube(s):
    return 0.7 * _t_reach3(s, T, O) + s[10] * _t_reach2(s, O, G)


# ---- 05 PushT -------------------------------------------------------
# r = ((cos(theta)+1)/2)^2 * 0.5 + (1 - tanh(5*d2))^2 * 0.5
# theta in feature 9
@bytecode
def gint_05_pusht(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _F(sblock, 9); fcos()
    faddimm(1.0); fmulimm(0.5); dup(); fmul(); fmulimm(0.5)   # 0.5*((cosθ+1)/2)^2
    _reach2(sblock, T, O); dup(); fmul(); fmulimm(0.5); fadd()
    fstg_1d(0, rblock); halt()


def torch_05_pusht(s):
    rot = ((torch.cos(s[9]) + 1.0) * 0.5) ** 2 * 0.5
    pos = _t_reach2(s, T, O) ** 2 * 0.5
    return rot + pos


# ---- 06 PokeCube ----------------------------------------------------
# r = reach3(tcp, obj) + (d3 < 0.04) * reach2(obj.xy, goal.xy)
@bytecode
def gint_06_pokecube(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O)
    _sq_dist3(sblock, T, O); fsqrt(); fpush(0.04); fgt()  # mask = 0.04>d
    _reach2(sblock, O, G); fmul(); fadd()
    fstg_1d(0, rblock); halt()


def torch_06_pokecube(s):
    reach = _t_reach3(s, T, O)
    d3 = torch.sqrt(_t_sq_dist3(s, T, O))
    mask = (d3 < 0.04).to(torch.float32)
    return reach + mask * _t_reach2(s, O, G)


# ---- 07 LiftPegUpright ---------------------------------------------
# r = |upright_dot| + (1-tanh(5*|peg_z - 0.5|)) + reach3(tcp,peg)/5
# upright_dot in feature 12, peg_z = obj_z (5)
@bytecode
def gint_07_lift_peg_upright(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    # |upright_dot| = sqrt(x^2)
    _F(sblock, 12); dup(); fmul(); fsqrt()
    # height term: 1 - tanh(5 * sqrt((peg_z - 0.5)^2))
    _F(sblock, 5); faddimm(-0.5); dup(); fmul(); fsqrt(); _inv_tanh_macro(5.0)
    fadd()
    _reach3(sblock, T, O); fmulimm(0.2); fadd()
    fstg_1d(0, rblock); halt()


def torch_07_lift_peg_upright(s):
    upright = torch.sqrt(s[12] * s[12])
    height = _t_inv_tanh(torch.sqrt((s[5] - 0.5) ** 2), 5.0)
    reach = _t_reach3(s, T, O) * 0.2
    return upright + height + reach


# ---- 08 RollBall ----------------------------------------------------
# r = reach3(tcp, obj, scale=2) + 5 * reach2(obj.xy, goal.xy, scale=1)
@bytecode
def gint_08_rollball(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O, scale=2.0)
    _reach2(sblock, O, G, scale=1.0); fmulimm(5.0); fadd()
    fstg_1d(0, rblock); halt()


def torch_08_rollball(s):
    return _t_reach3(s, T, O, 2.0) + 5.0 * _t_reach2(s, O, G, 1.0)


# ---- 09 PlaceSphere -------------------------------------------------
# r = reach3(tcp, obj) + g * reach3(obj, bin) + p * (1 - tanh(10*qvel))
# bin = goal slot, qvel = scalar_a (9)
@bytecode
def gint_09_placesphere(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O)
    _F(sblock, 10); _reach3(sblock, O, G); fmul(); fadd()
    _F(sblock, 11); _F(sblock, 9); _inv_tanh_macro(10.0); fmul(); fadd()
    fstg_1d(0, rblock); halt()


def torch_09_placesphere(s):
    return (_t_reach3(s, T, O)
            + s[10] * _t_reach3(s, O, G)
            + s[11] * _t_inv_tanh(s[9], 10.0))


# ---- 10 PegInsertSide ----------------------------------------------
# r = reach3(tcp, peg, scale=4) + g * 3 * reach3(peg_head, hole, scale=4.5)
# peg_head = goal_xyz (6..8) treated as the head, hole = scalar_b..d (10..12)
@bytecode
def gint_10_peginsert(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O, scale=4.0)
    _F(sblock, 13)                              # g (placed in scalar_e)
    _reach3(sblock, G, 10, scale=4.5); fmulimm(3.0)
    fmul(); fadd()
    fstg_1d(0, rblock); halt()


def torch_10_peginsert(s):
    return _t_reach3(s, T, O, 4.0) + s[13] * 3.0 * _t_reach3(s, G, 10, 4.5)


# ---- 11 PullCubeTool -----------------------------------------------
# r = 2*reach3(tcp,tool) + 2*g + 1.5*reach3(hook, target, 3.0)
@bytecode
def gint_11_pullcubetool(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O); fmulimm(2.0)
    _F(sblock, 10); fmulimm(2.0); fadd()
    _reach3(sblock, O, G, scale=3.0); fmulimm(1.5); fadd()
    fstg_1d(0, rblock); halt()


def torch_11_pullcubetool(s):
    return (2.0 * _t_reach3(s, T, O)
            + 2.0 * s[10]
            + 1.5 * _t_reach3(s, O, G, 3.0))


# ---- 12 TwoRobotPickCube -------------------------------------------
# Two TCPs (tcp_L = features 0..2, tcp_R = features 6..8) reach for obj (3..5)
# r = 0.5 * (reach3(tcp_L,obj) + reach3(tcp_R,obj))  -- symmetric dual-arm
# (we reuse goal_xyz slots as tcp_R since the layout is shared)
@bytecode
def gint_12_tworobot(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O)
    _reach3(sblock, G, O); fadd()
    fmulimm(0.5)
    fstg_1d(0, rblock); halt()


def torch_12_tworobot(s):
    return 0.5 * (_t_reach3(s, T, O) + _t_reach3(s, G, O))


# ---- 13 PickSingleYCB ----------------------------------------------
# r = reach3(tcp,obj) + g + g*reach3(obj,goal) + g*p
@bytecode
def gint_13_picksingleycb(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O)
    _F(sblock, 10); fadd()
    _F(sblock, 10); _reach3(sblock, O, G); fmul(); fadd()
    _F(sblock, 10); _F(sblock, 11); fmul(); fadd()
    fstg_1d(0, rblock); halt()


def torch_13_picksingleycb(s):
    g = s[10]; p = s[11]
    return _t_reach3(s, T, O) + g + g * _t_reach3(s, O, G) + g * p


# ---- 14 CartpoleBalance --------------------------------------------
# r = upright * centered * small_ctrl * small_vel
# theta = 9, x = 10, action = 11, omega = 12
@bytecode
def gint_14_cartpole(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _F(sblock, 9); fcos(); faddimm(1.0); fmulimm(0.5)         # upright
    _F(sblock, 10); dup(); fmul(); fsqrt(); _inv_tanh_macro(1.0)  # centered
    fmul()
    _F(sblock, 11); dup(); fmul(); fmulimm(-0.2); faddimm(1.0)    # small_ctrl
    fmul()
    _F(sblock, 12); dup(); fmul(); fsqrt(); _inv_tanh_macro(1.0)  # small_vel
    fmul()
    fstg_1d(0, rblock); halt()


def torch_14_cartpole(s):
    upright = (torch.cos(s[9]) + 1.0) * 0.5
    centered = _t_inv_tanh(torch.sqrt(s[10] ** 2), 1.0)
    small_ctrl = 1.0 - 0.2 * s[11] ** 2
    small_vel = _t_inv_tanh(torch.sqrt(s[12] ** 2), 1.0)
    return upright * centered * small_ctrl * small_vel


# ---- 15 HopperHop --------------------------------------------------
# r = (1 - tanh(|h - 1.0|)) * (1 + tanh(v_x)) / 2
# h = scalar_a (9), v_x = scalar_b (10)
@bytecode
def gint_15_hopper(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _F(sblock, 9); faddimm(-1.0); dup(); fmul(); fsqrt(); _inv_tanh_macro(1.0)
    _F(sblock, 10); _tanh_macro(1.0); faddimm(1.0); fmulimm(0.5)
    fmul()
    fstg_1d(0, rblock); halt()


def torch_15_hopper(s):
    height = _t_inv_tanh(torch.sqrt((s[9] - 1.0) ** 2), 1.0)
    speed = (1.0 + _t_tanh(s[10], 1.0)) * 0.5
    return height * speed


# ---- 16 AnymalLocomotion -------------------------------------------
# r = exp(-(v_x - v*)^2 / 0.25) + 0.5*exp(-(omega - omega*)^2 / 0.25) - 0.05*torque^2
# v_x=9, v*=10, omega=11, omega*=12, torque=13
@bytecode
def gint_16_anymal(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    # exp(-(v_x - v*)^2 * 4)
    _F(sblock, 9); _F(sblock, 10); fsub(); dup(); fmul()
    fmulimm(-4.0); fexp()
    # + 0.5 * exp(-(omega - omega*)^2 * 4)
    _F(sblock, 11); _F(sblock, 12); fsub(); dup(); fmul()
    fmulimm(-4.0); fexp(); fmulimm(0.5); fadd()
    # - 0.05 * torque^2
    _F(sblock, 13); dup(); fmul(); fmulimm(-0.05); fadd()
    fstg_1d(0, rblock); halt()


def torch_16_anymal(s):
    track_v = torch.exp(-(s[9] - s[10]) ** 2 * 4.0)
    track_w = 0.5 * torch.exp(-(s[11] - s[12]) ** 2 * 4.0)
    pen = 0.05 * s[13] ** 2
    return track_v + track_w - pen


# ---- 17 RotateValve ------------------------------------------------
# r = (1 - tanh(10 * err)) + 4 * tanh(0.5 * omega)
# err = scalar_a (9), omega = scalar_b (10)
@bytecode
def gint_17_rotatevalve(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _F(sblock, 9); _inv_tanh_macro(10.0)
    _F(sblock, 10); _tanh_macro(0.5); fmulimm(4.0); fadd()
    fstg_1d(0, rblock); halt()


def torch_17_rotatevalve(s):
    return _t_inv_tanh(s[9], 10.0) + 4.0 * _t_tanh(s[10], 0.5)


# ---- 18 RotateCubeInHand -------------------------------------------
# r = 20*angle - 0.1*v - 50*fall - 0.0003*power
# angle=9, v=10, fall=11, power=12
@bytecode
def gint_18_rotatecube(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _F(sblock, 9); fmulimm(20.0)
    _F(sblock, 10); fmulimm(-0.1); fadd()
    _F(sblock, 11); fmulimm(-50.0); fadd()
    _F(sblock, 12); fmulimm(-3e-4); fadd()
    fstg_1d(0, rblock); halt()


def torch_18_rotatecube(s):
    return 20.0 * s[9] - 0.1 * s[10] - 50.0 * s[11] - 3e-4 * s[12]


# ---- 19 OpenCabinetDrawer ------------------------------------------
# r = reach3(tcp, handle); if frac>0.999: r = 2; r += 2*frac
# frac = scalar_a (9)
@bytecode
def gint_19_cabinet(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O)
    # mask: frac > 0.999 ? 2 : reach
    fpush(2.0); _F(sblock, 9); fpush(0.999); flt()  # mask = 0.999<frac
    fselect()
    _F(sblock, 9); fmulimm(2.0); fadd()
    fstg_1d(0, rblock); halt()


def torch_19_cabinet(s):
    reach = _t_reach3(s, T, O)
    base = torch.where(s[9] > 0.999, torch.full_like(reach, 2.0), reach)
    return base + 2.0 * s[9]


# ---- 20 AssemblingKits ---------------------------------------------
# r = reach3(part, slot) + 0.5 * (1 + |q_dot|) - 0.01 * action_norm^2
# q_dot = scalar_a (9), action_norm = scalar_b (10)
@bytecode
def gint_20_assembling(state, reward, REGW, WARP):
    sblock = make_block_1d(state, NUM_FEATURES * NUM_SCENES, 1)
    rblock = make_block_1d(reward, NUM_SCENES, 1)
    _reach3(sblock, T, O)
    _F(sblock, 9); dup(); fmul(); fsqrt()       # |q_dot|
    faddimm(1.0); fmulimm(0.5); fadd()
    _F(sblock, 10); dup(); fmul(); fmulimm(-0.01); fadd()
    fstg_1d(0, rblock); halt()


def torch_20_assembling(s):
    qabs = torch.sqrt(s[9] ** 2)
    return (_t_reach3(s, T, O)
            + 0.5 * (1.0 + qabs)
            - 0.01 * s[10] ** 2)


# Catalog of (name, gint program, torch reference) tuples in launch order.
TASKS = [
    ('PickCube',         gint_01_pickcube,         torch_01_pickcube),
    ('StackCube',        gint_02_stackcube,        torch_02_stackcube),
    ('PushCube',         gint_03_pushcube,         torch_03_pushcube),
    ('PullCube',         gint_04_pullcube,         torch_04_pullcube),
    ('PushT',            gint_05_pusht,            torch_05_pusht),
    ('PokeCube',         gint_06_pokecube,         torch_06_pokecube),
    ('LiftPegUpright',   gint_07_lift_peg_upright, torch_07_lift_peg_upright),
    ('RollBall',         gint_08_rollball,         torch_08_rollball),
    ('PlaceSphere',      gint_09_placesphere,      torch_09_placesphere),
    ('PegInsertSide',    gint_10_peginsert,        torch_10_peginsert),
    ('PullCubeTool',     gint_11_pullcubetool,     torch_11_pullcubetool),
    ('TwoRobotPickCube', gint_12_tworobot,         torch_12_tworobot),
    ('PickSingleYCB',    gint_13_picksingleycb,    torch_13_picksingleycb),
    ('CartpoleBalance',  gint_14_cartpole,         torch_14_cartpole),
    ('HopperHop',        gint_15_hopper,           torch_15_hopper),
    ('AnymalLocomotion', gint_16_anymal,           torch_16_anymal),
    ('RotateValve',      gint_17_rotatevalve,      torch_17_rotatevalve),
    ('RotateCubeInHand', gint_18_rotatecube,       torch_18_rotatecube),
    ('OpenCabinet',      gint_19_cabinet,          torch_19_cabinet),
    ('AssemblingKits',   gint_20_assembling,       torch_20_assembling),
]

assert len(TASKS) == NUM_TASKS
