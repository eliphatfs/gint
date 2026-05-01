"""
FX Graph to gint bytecode compiler.

Converts PyTorch FX graphs into gint stack-machine bytecode programs.

Architecture
------------
* GraphPartitioner  – splits the FX graph into subgraphs that fit within gint's
                      hardware constraints (max tensors, max stack depth, same shape,
                      supported ops).
* ForestTraverser   – turns a DAG subgraph into a post-order schedule for the
                      stack machine.
* StackCodegen      – emits bytecode for a scheduled subgraph, managing a virtual
                      stack and calling frontend module functions so that all
                      instruction emission goes through the same path as hand-written
                      sugar programs.
* GintCompiler      – orchestrates the above to produce an executable callable.
"""

import math
import torch
import numpy as np
from typing import Callable, Dict, List, Optional, Set, Union
from torch.fx import GraphModule, Node

from ..host.executor import (
    BaseExecutableProgram,
    ProgramData,
    ProgramTensorInfo,
    TensorInterface,
)
from ..host import frontend as fe
from ..host.frontend import FrontendState, _frontend_state
from .op_registry import (
    OpDescriptor, OpKind, get_op_descriptor, resolve_arg_order,
    _emit_width_combine,
)


# ---------------------------------------------------------------------------
# Broadcasting utilities
# ---------------------------------------------------------------------------

def _broadcast_shapes(*shapes):
    """Compute NumPy-style broadcast output shape. Returns None if incompatible."""
    if not shapes:
        return ()
    ndim = max(len(s) for s in shapes)
    result = []
    for i in range(ndim):
        dims = []
        for s in shapes:
            idx = i - (ndim - len(s))
            dims.append(s[idx] if idx >= 0 else 1)
        max_dim = max(dims)
        if any(d != 1 and d != max_dim for d in dims):
            return None
        result.append(max_dim)
    return tuple(result)


def _c_contiguous_strides(shape):
    """Compute C-contiguous strides for a shape."""
    if not shape:
        return []
    strides = []
    prod = 1
    for s in reversed(shape):
        strides.append(prod)
        prod *= s
    return list(reversed(strides))


def _merge_dims(shape, strides_list):
    """Greedy-merge consecutive dims that are C-contiguous in *every* tensor.

    For each adjacent pair (i, i+1), the dims are merged iff
    ``stride[i] == stride[i+1] * shape[i+1]`` holds for *all* tensors in
    ``strides_list``. Merging continues until no more pairs qualify.

    Returns ``(merged_shape, [merged_strides_per_tensor])``. Unlike the all-
    or-nothing merge in _compute_broadcast_plan, this is incremental — a
    non-contiguous boundary stops the merge there but doesn't collapse it
    back to length-1.
    """
    if len(shape) == 0:
        return tuple(), [tuple() for _ in strides_list]
    if len(shape) == 1:
        return tuple(shape), [tuple(s) for s in strides_list]

    out_shape = list(shape)
    out_strides = [list(s) for s in strides_list]

    i = 0
    while i + 1 < len(out_shape):
        can_merge = all(
            s[i] == s[i + 1] * out_shape[i + 1]
            for s in out_strides
        )
        if can_merge:
            out_shape[i] = out_shape[i] * out_shape[i + 1]
            del out_shape[i + 1]
            for s in out_strides:
                s[i] = s[i + 1]
                del s[i + 1]
        else:
            i += 1

    return tuple(out_shape), [tuple(s) for s in out_strides]


def _select_reduction_tiling(N: int) -> dict:
    """Select reduction tile parameters based on innermost-dim size N.

    Returns a dict with::
        B    – batches handled per warp (B * R == 128)
        R    – reduction elements per chunk-load
        mode – '1d', '2dt', or '2dw' (matches the load instruction family)
        do_warp_reduce  – emit warp_allreduce after the chunk loop
        do_width_combine – emit the 4-wide width-lane combine sequence
        warp_axis  – which load-tile dim ('thread' or 'width') carries
                     the *reduction* axis (drives stride placement)
        batch_axis – which load-tile dim carries the *batch* axis

    The four cases (B, R, mode):
      N >= 128 → (1, 128, '1d')   threads+width flatten over reduction
      16-127   → (4, 32,  '2dt')  threads scan reduction, width carries batches
      <16      → (32, 4,  '2dw')  threads carry batches, width scans reduction
    """
    if N >= 128:
        return dict(B=1, R=128, mode='1d',
                    do_warp_reduce=True, do_width_combine=True,
                    warp_axis='thread', batch_axis=None)
    if N >= 16:
        return dict(B=4, R=32, mode='2dt',
                    do_warp_reduce=True, do_width_combine=False,
                    warp_axis='thread', batch_axis='width')
    return dict(B=32, R=4, mode='2dw',
                do_warp_reduce=False, do_width_combine=True,
                warp_axis='width', batch_axis='thread')


def _select_pointwise_tiling(block_size: int) -> dict:
    """Select pointwise tile parameters based on the merged inner-block size.

    Mirrors `_select_reduction_tiling` but adapted for pointwise — no warp
    reduction, just per-tile lane packing. Same B*R=128 budget, same three
    modes. Choice trades off two things:

      - lane utilization (high = useful work per warp)
      - memory coalescing for C-contiguous (M, K) inputs (consecutive
        threads reading consecutive addresses)

    Tier choice:
      block_size >= 128 → (1, 128, '1d')   1 batch/warp scanning K per warp;
                                           threads stride 1 → coalesced
      16-127            → (4, 32,  '2dt')  4 batches × ≤32 inner per warp;
                                           threads stride 1 in inner → coalesced
      < 16              → (32, 4,  '2dw')  32 batches × ≤4 inner per warp;
                                           threads stride K (small) → ~K cache
                                           lines per width-lane, amortized over
                                           32× fewer warps

    For block_size < 128 with the existing flat 1d path, lane utilization
    is `block_size/128`; this dispatch lifts it toward 100% (modulo non-
    multiple-of-R remainders) at the cost of strided thread access.
    """
    if block_size >= 128:
        return dict(B=1, R=128, mode='1d')
    if block_size >= 16:
        return dict(B=4, R=32, mode='2dt')
    return dict(B=32, R=4, mode='2dw')


def _compute_broadcast_plan(output_shape, tensor_shapes, tensor_strides=None):
    """Compute a broadcast plan mapping output_shape to gint block+batch decomposition.

    If *tensor_strides* is provided (parallel list to *tensor_shapes*), actual
    memory strides are used instead of computing C-contiguous strides from shape.

    Returns a plan dict with 'block_size', 'batch_dims', 'per_tensor', 'grid_dim',
    'num_inner_blocks', or None if infeasible (>4 batch dims).
    """
    ndim = len(output_shape)
    if ndim == 0:
        return None

    # Right-align all tensor shapes (pad with 1s on the left)
    padded = []
    for s in tensor_shapes:
        pad = ndim - len(s)
        padded.append((1,) * pad + tuple(s))

    # Validate broadcast compatibility: each dim must be 1 or equal to output
    for p in padded:
        for d in range(ndim):
            if p[d] != 1 and p[d] != output_shape[d]:
                return None

    # Merge consecutive innermost dims where ALL tensors match output (non-broadcast)
    merge_count = 0
    for d in range(ndim - 1, -1, -1):
        if all(p[d] == output_shape[d] for p in padded):
            merge_count += 1
        else:
            break
    # Always include at least the innermost dim
    if merge_count == 0:
        merge_count = 1

    # If actual strides are non-contiguous in the merged inner dims, we can't
    # safely merge — the kernel's flat block access requires the merged region
    # to be contiguous in memory.  Clamp merge_count back to 1.
    if tensor_strides is not None:
        for p, orig_strides in zip(padded, tensor_strides):
            if orig_strides is None:
                continue
            c_orig = _c_contiguous_strides(p[-len(orig_strides):])
            merge_in_orig = min(merge_count, len(orig_strides))
            if tuple(orig_strides[-merge_in_orig:]) != tuple(c_orig[-merge_in_orig:]):
                merge_count = 1
                break

    block_size = 1
    for d in range(ndim - merge_count, ndim):
        block_size *= output_shape[d]

    batch_dims = list(output_shape[:ndim - merge_count])
    if len(batch_dims) > 4:
        return None

    num_inner_blocks = max(1, (block_size + 127) // 128)
    batch_product = 1
    for d in batch_dims:
        batch_product *= d
    grid_dim = num_inner_blocks * batch_product

    per_tensor = []
    for i, p in enumerate(padded):
        orig_strides = tensor_strides[i] if tensor_strides is not None else None
        pad_ndim = ndim - (len(orig_strides) if orig_strides is not None else 0)
        c_strides = _c_contiguous_strides(p)

        # block_stride: 0 if any merged inner dim broadcasts, else actual
        # innermost-dim stride (or 1 for C-contiguous fallback)
        inner_broadcast = any(
            p[d] == 1 and output_shape[d] > 1
            for d in range(ndim - merge_count, ndim)
        )
        if inner_broadcast:
            block_stride = 0
        elif orig_strides is not None:
            block_stride = orig_strides[-1]
        else:
            block_stride = 1

        # batch_strides: 0 for broadcast dims, actual stride otherwise
        b_strides = []
        for d in range(len(batch_dims)):
            if p[d] == 1 and output_shape[d] > 1:
                b_strides.append(0)
            elif orig_strides is not None and d >= pad_ndim:
                b_strides.append(orig_strides[d - pad_ndim])
            else:
                b_strides.append(c_strides[d])

        per_tensor.append({
            'block_stride': block_stride,
            'batch_strides': b_strides,
        })

    return {
        'block_size': block_size,
        'batch_dims': batch_dims,
        'per_tensor': per_tensor,
        'grid_dim': grid_dim,
        'num_inner_blocks': num_inner_blocks,
    }


# ---------------------------------------------------------------------------
# Compiled-subgraph wrapper
# ---------------------------------------------------------------------------

class GintCompiledSubgraph(BaseExecutableProgram):
    """An already-compiled gint subgraph ready for execution."""

    def __init__(self, bytecode: List[List[int]], tensor_infos: List[ProgramTensorInfo],
                 output_shape: tuple = (), grid_dim: int = 1,
                 input_adjustments: List[tuple] = None):
        """
        *input_adjustments*: list of (global_idx, dim, start) for slice inputs
        whose base pointer needs adjustment at runtime.
        """
        super().__init__()
        self.bytecode = bytecode
        self.tensor_infos = tensor_infos
        self.output_shape = output_shape
        self.grid_dim = grid_dim
        self.input_adjustments = input_adjustments or []

    def get_program(self, *args: TensorInterface, **extra_kwargs) -> ProgramData:
        bc_array = np.array(self.bytecode, dtype=np.int32).reshape(-1)
        return ProgramData(bc_array, self.tensor_infos)


# ---------------------------------------------------------------------------
# DAG scheduling
# ---------------------------------------------------------------------------

class ForestTraverser:
    """Post-order traversal of a DAG subgraph, producing a linear schedule."""

    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.node_set = set(nodes)

    def get_schedule(self) -> List[Node]:
        # Roots: nodes whose users are all outside the subgraph (or have no in-subgraph users).
        roots = [n for n in self.nodes
                 if not any(u in self.node_set for u in n.users)]

        visited: Set[Node] = set()
        schedule: List[Node] = []

        def visit(n: Node):
            if n in visited or n not in self.node_set:
                return
            for arg in n.args:
                if isinstance(arg, Node):
                    visit(arg)
            visited.add(n)
            schedule.append(n)

        for root in roots:
            visit(root)
        return schedule


# ---------------------------------------------------------------------------
# Register-spill planner
# ---------------------------------------------------------------------------

# Hardware budget for the l12 kernel variant: POOL_SIZE=12 slots are shared
# between stack and registers. Stack occupies pool[0..7], reg N occupies
# pool[POOL_SIZE-1-N]. We keep ``stack_depth + active_regs <= 11`` so there
# is one slot of headroom for transient ``dup``/``swap`` shuffling.
_SPILL_MAX_STACK = 8
_SPILL_NUM_REGS  = 8
_SPILL_POOL_LIMIT = 11


def _plan_spills(
    nodes: List[Node],
    ext_io: Set[Node],
    max_stack: int = _SPILL_MAX_STACK,
    num_regs: int = _SPILL_NUM_REGS,
    pool_limit: int = _SPILL_POOL_LIMIT,
):
    """Decide which multi-user intermediates should live in virtual registers.

    Walks the schedule and assigns a virtual register to every node with
    >1 distinct downstream user inside the subgraph (i.e. values that
    would otherwise need a global-memory round-trip to handle the
    "buried at depth ≥ 2" reload case in ``StackCodegen.handle_operand``).

    Registers are reused across non-overlapping live ranges via a simple
    Belady-style allocator: free registers whose owner just died, then
    allocate the smallest free index for the new result. If the live
    multi-user set ever exceeds ``num_regs``, the overflow nodes go back
    to a global-tensor slot (returned in ``overflow``).

    Concurrently simulates the abstract stack to verify pool feasibility:
      - stack_peak (during op emission) ≤ max_stack
      - on_stack_live + active_regs ≤ pool_limit (slot conflicts in the
        pool happen otherwise — stack and registers share pool[0..11]).

    Returns ``(node_to_reg, overflow, feasible)``."""
    node_set = set(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Distinct downstream users inside the subgraph + last-use index.
    later_user_count: Dict[Node, int] = {}
    last_use: Dict[Node, int] = {}
    for n in nodes:
        users_after = [u for u in n.users
                       if u in node_set and node_idx[u] > node_idx[n]]
        later_user_count[n] = len(users_after)
        if users_after:
            last_use[n] = max(node_idx[u] for u in users_after)

    # External-IO nodes with at least one in-subgraph consumer also occupy
    # a stack slot until that last consumer (codegen does ``dup; store``
    # so a copy stays on stack). Track them like any other live value.
    candidate_set: Set[Node] = {
        n for n in nodes
        if later_user_count.get(n, 0) > 1 and n not in ext_io
    }

    free_regs = list(range(num_regs))
    in_reg: Dict[Node, int] = {}
    node_to_reg: Dict[Node, int] = {}
    overflow: Set[Node] = set()
    on_stack_live: Set[Node] = set()
    feasible = True

    for i, node in enumerate(nodes):
        op_desc = get_op_descriptor(node)
        arity = op_desc.arity if op_desc else 0
        peak_extra = op_desc.peak_stack_extra if op_desc else 0

        # Peak during emission: pre-existing on-stack live + freshly pushed
        # args + op's internal peak. Conservative — every arg push counts
        # as +1 even when the codegen could skip the dup at depth 0.
        peak = len(on_stack_live) + arity + peak_extra
        if peak > max_stack:
            feasible = False
        if peak + len(in_reg) > pool_limit:
            feasible = False

        # Free dying args (last use at this step). Args may be in regs,
        # on the stack-live set, or both for the rare ext-IO case.
        for arg in node.args:
            if not isinstance(arg, Node):
                continue
            if last_use.get(arg, -1) != i:
                continue
            if arg in in_reg:
                free_regs.append(in_reg.pop(arg))
                free_regs.sort()
            if arg in on_stack_live:
                on_stack_live.discard(arg)

        # Result placement.
        has_internal_use = later_user_count.get(node, 0) > 0
        if not has_internal_use:
            # Stored to global (ext_io) or popped immediately. No slot.
            pass
        elif node in candidate_set:
            if free_regs:
                reg = free_regs.pop(0)
                in_reg[node] = reg
                node_to_reg[node] = reg
            else:
                overflow.add(node)
                on_stack_live.add(node)
        else:
            # Single-user internal, OR ext_io with internal uses.
            on_stack_live.add(node)

        if len(on_stack_live) > max_stack:
            feasible = False
        if len(on_stack_live) + len(in_reg) > pool_limit:
            feasible = False

    return node_to_reg, overflow, feasible


# ---------------------------------------------------------------------------
# Virtual-stack code generator
# ---------------------------------------------------------------------------

class StackCodegen:
    """
    Tracks a virtual stack of FX nodes and emits bytecode via the frontend module.

    All fe.* calls are routed through the active _frontend_state context, so the
    compiler reuses exactly the same instruction-encoding path as hand-written
    sugar programs.
    """

    def __init__(self, max_stack: int, tensor_map: Dict[Node, int],
                 load_fn: Optional[Callable] = None,
                 store_fn: Optional[Callable] = None,
                 node_to_reg: Optional[Dict[Node, int]] = None):
        self.max_stack = max_stack
        self.tensor_map = tensor_map
        # Tile-aware load/store entry points. Default to fldg_1d/fstg_1d
        # (the existing flat-1d path). Pointwise tile dispatch (2dt/2dw)
        # passes the matching frontend functions; the rest of the codegen
        # is mode-agnostic since pointwise ops act per-lane.
        self.load_fn = load_fn or fe.fldg_1d
        self.store_fn = store_fn or fe.fstg_1d
        # Virtual stack: list of FX nodes (or None for constants), bottom first.
        self.vstack: List[Optional[Node]] = []
        # Remaining internal uses for each node (decremented as args are consumed).
        # Counts arg-appearances (multiset), so mul(x, x) counts x twice.
        self.uses_left: Dict[Node, int] = {}
        # Nodes that must be stored to global memory (external outputs or
        # multi-use intermediates already registered in tensor_map).
        self.ext_out: Set[Node] = set()
        # How many times each Node has been claimed for the current op's
        # operand list. Reset at the end of emit_op. Lets handle_operand
        # distinguish "top is mine from a prior op" (no dup needed for
        # depth==0) from "top is the slot I just pushed for arg #i, and
        # arg #i+1 wants the same value" (must dup).
        self._current_op_pushed: Dict[Node, int] = {}
        # Spill plan: nodes mapped here go to virtual registers instead of
        # staying on the stack. The codegen emits ``fstore_reg(N)`` after
        # computing such a node, and ``fload_reg(N)`` to load it as an
        # operand.
        self.node_to_reg: Dict[Node, int] = node_to_reg or {}

    def set_subgraph_nodes(self, nodes: List[Node], graph_outputs: Set[Node]):
        node_set = set(nodes)
        self.uses_left = {}
        self.ext_out = set()
        # Count internal uses by arg-appearance, not unique-user count.
        # node.users is a dict keyed on user nodes, so mul(x, x) reports
        # x.users == [mul] even though x is consumed twice.
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, Node) and arg in node_set:
                    self.uses_left[arg] = self.uses_left.get(arg, 0) + 1
        for node in nodes:
            self.uses_left.setdefault(node, 0)
            if node in graph_outputs or any(u not in node_set for u in node.users):
                self.ext_out.add(node)

    # --- Virtual-stack helpers ---

    def _depth(self, node: Node) -> int:
        """Depth of node from top (0 = top).  Returns -1 if not on stack."""
        for i, n in enumerate(reversed(self.vstack)):
            if n is node:
                return i
        return -1

    def _check_overflow(self):
        if len(self.vstack) > self.max_stack:
            raise RuntimeError(f"Stack overflow: depth {len(self.vstack)} > {self.max_stack}")

    # --- Operand handling ---

    def handle_operand(self, arg: Union[Node, int, float]):
        """
        Ensure *arg* is on top of the virtual stack, emitting bytecode as needed.
        Also decrements uses_left for Node args so post_op sees the correct count.
        """
        if not isinstance(arg, Node):
            # Scalar constant → LoadImm
            fe.fpush(float(arg))
            self.vstack.append(None)
            self._check_overflow()
            return

        uses_left = self.uses_left.get(arg, 0)
        depth = self._depth(arg)
        already_claimed = self._current_op_pushed.get(arg, 0)
        self._current_op_pushed[arg] = already_claimed + 1

        if depth == -1:
            # Not on virtual stack: load from register (if spilled there)
            # or from global memory. Register loads are 1 instruction and
            # don't go through L1/L2; preferred over a global reload.
            if arg in self.node_to_reg:
                fe.fload_reg(self.node_to_reg[arg])
                self.vstack.append(arg)
            elif arg in self.tensor_map:
                self.load_fn(0, self.tensor_map[arg])
                self.vstack.append(arg)
            else:
                raise RuntimeError(f"Node {arg.name} not on stack, not in register, and has no global tensor entry")

        elif depth == 0:
            # Already on top. Duplicate when:
            # - the top slot is already claimed by an earlier operand of the
            #   current op (e.g. mul(x, x) — the second x needs its own slot), or
            # - more uses remain after this one (preserve a copy for later).
            if already_claimed >= 1 or uses_left > 1:
                fe.dup()
                self.vstack.append(arg)

        elif depth == 1:
            # One slot below top.
            if arg in self.node_to_reg:
                # A reg-load is cheaper than reshuffling the stack.
                fe.fload_reg(self.node_to_reg[arg])
                self.vstack.append(arg)
            elif arg in self.tensor_map:
                # Cheaper to reload from global than shuffle the stack (avoids a
                # Swap that would displace the current top for later ops).
                self.load_fn(0, self.tensor_map[arg])
                self.vstack.append(arg)
            else:
                # Bring to top via Swap.
                fe.swap()
                top  = self.vstack.pop()   # element that was on top
                sec  = self.vstack.pop()   # arg (depth 1)
                self.vstack.append(top)    # top sinks to depth 1
                self.vstack.append(sec)    # arg rises to top

                if uses_left > 1:
                    # Keep a copy for future uses: DupX1 inserts a copy of top below second.
                    # Before DupX1: [..., top, arg]  top=arg
                    # DupX1: v1=arg(top), v2=top(2nd) -> pop2 push(arg) push(top) push(arg)
                    #        -> [..., arg, top, arg]  top=arg
                    fe.dupx1()
                    v1 = self.vstack.pop()   # arg
                    v2 = self.vstack.pop()   # top (the other element)
                    self.vstack.append(v1)   # buried copy for future use
                    self.vstack.append(v2)   # other element
                    self.vstack.append(v1)   # top copy consumed by current op

        else:
            # Depth >= 2: reload from register or global if possible.
            if arg in self.node_to_reg:
                fe.fload_reg(self.node_to_reg[arg])
                self.vstack.append(arg)
            elif arg in self.tensor_map:
                self.load_fn(0, self.tensor_map[arg])
                self.vstack.append(arg)
            else:
                raise RuntimeError(
                    f"Node {arg.name} buried at depth {depth} with no global tensor entry; "
                    "subgraph scheduling produced an unresolvable stack state"
                )

        self._check_overflow()

        # Consume one internal use.
        if arg in self.uses_left:
            self.uses_left[arg] = max(0, uses_left - 1)

    # --- Operation emission ---

    def emit_op(self, node: Node, op_desc: OpDescriptor, num_pushed: Optional[int] = None):
        """
        Emit the operation, update the virtual stack (pop inputs, push result),
        then handle result storage / cleanup.

        ``num_pushed`` is how many items the caller actually pushed for this
        op (which may differ from ``op_desc.arity`` when the descriptor folds
        a scalar arg into an immediate instruction and skips its push).
        Defaults to ``op_desc.arity`` for callers that don't track it.
        """
        if num_pushed is None:
            num_pushed = op_desc.arity

        # Emit the instruction(s) for this operation.
        op_desc.emit_fn(node)

        # Pop consumed inputs from the virtual stack.
        for _ in range(num_pushed):
            self.vstack.pop()

        # Push result.
        self.vstack.append(node)

        # Handle result: store to global, store to register, and/or keep/discard
        # from the virtual stack. The order matters because each store consumes
        # the top of stack; ``dup`` is used to fan out to multiple destinations.
        internal_uses = self.uses_left.get(node, 0)
        in_global     = node in self.tensor_map
        in_reg        = node in self.node_to_reg

        if in_global:
            # External outputs go to global. If they ALSO need a register
            # (rare — partitioner usually reloads from global instead), we'd
            # need a dup. ``_plan_spills`` skips tensor_map nodes so this
            # branch shouldn't fire, but keep it explicit for safety.
            if in_reg:
                fe.dup()
                self.vstack.append(node)
                fe.fstore_reg(self.node_to_reg[node])
                self.vstack.pop()
            if internal_uses > 0:
                # Keep on stack for upcoming internal consumers.
                fe.dup()
                self.vstack.append(node)
                self.store_fn(0, self.tensor_map[node])
                self.vstack.pop()
            else:
                self.store_fn(0, self.tensor_map[node])
                self.vstack.pop()
        elif in_reg:
            # Spill to register. The store_reg pops the top of stack.
            fe.fstore_reg(self.node_to_reg[node])
            self.vstack.pop()
        elif internal_uses == 0:
            # Result is not needed by anyone: discard.
            fe.pop()
            self.vstack.pop()
        # else: internal_uses > 0, result stays on the virtual stack.

        # Reset per-op claims; next op starts with no claimed slots.
        self._current_op_pushed = {}


# ---------------------------------------------------------------------------
# Graph partitioner
# ---------------------------------------------------------------------------

class GraphPartitioner:
    """
    Partitions an FX graph into subgraphs compatible with gint.

    Constraints (per subgraph):
    - All nodes use a supported op (registered in OP_REGISTRY).
    - All tensor nodes have the same shape.
    - At most *max_tensors* global tensor slots (external inputs + outputs +
      multi-use intermediates).
    - Stack depth never exceeds *max_stack* during execution.
    """

    def __init__(self, gm: GraphModule, max_tensors: int = 8, max_stack: int = 8):
        self.gm = gm
        self.max_tensors = max_tensors
        self.max_stack = max_stack

        self.graph_outputs: Set[Node] = set()
        for node in gm.graph.nodes:
            if node.op == 'output':
                for arg in node.args:
                    if isinstance(arg, Node):
                        self.graph_outputs.add(arg)
                    elif isinstance(arg, (list, tuple)):
                        for a in arg:
                            if isinstance(a, Node):
                                self.graph_outputs.add(a)

    def _get_shape(self, node: Node):
        if 'tensor_meta' in node.meta:
            return node.meta['tensor_meta'].shape
        if 'val' in node.meta and hasattr(node.meta['val'], 'shape'):
            return node.meta['val'].shape
        return None

    @staticmethod
    def _get_strides(node: Node):
        """Read actual strides from FX metadata, falling back to C-contiguous."""
        if 'val' in node.meta and hasattr(node.meta['val'], 'stride'):
            return node.meta['val'].stride()
        shape = None
        if 'tensor_meta' in node.meta:
            shape = node.meta['tensor_meta'].shape
        elif 'val' in node.meta and hasattr(node.meta['val'], 'shape'):
            shape = node.meta['val'].shape
        if shape is not None:
            return _c_contiguous_strides(shape)
        return None

    def _is_supported(self, node: Node) -> bool:
        if node.op != 'call_function':
            return False
        return get_op_descriptor(node) is not None

    @staticmethod
    def _is_slice_node(node: Node) -> bool:
        op_desc = get_op_descriptor(node)
        return (op_desc is not None and op_desc.kind == OpKind.METADATA
                and node.target == torch.ops.aten.slice.Tensor)

    @staticmethod
    def _effective_global_shapes(nodes, global_nodes, raw_shapes):
        """Replace global shapes with effective shapes for slice inputs.

        When a slice node is in *nodes*, its input global should use the
        slice OUTPUT shape (the only part the subgraph accesses).
        """
        result = list(raw_shapes)
        node_set = set(nodes)
        for node in nodes:
            if GraphPartitioner._is_slice_node(node):
                inp = node.args[0]
                if isinstance(inp, Node) and inp in global_nodes:
                    idx = global_nodes.index(inp)
                    out_shape = None
                    if 'tensor_meta' in node.meta:
                        out_shape = node.meta['tensor_meta'].shape
                    elif 'val' in node.meta and hasattr(node.meta['val'], 'shape'):
                        out_shape = node.meta['val'].shape
                    if out_shape is not None:
                        result[idx] = out_shape
        return result

    def partition(self) -> List[List[Node]]:
        subgraphs: List[List[Node]] = []
        current: List[Node] = []
        current_shape = None

        for node in self.gm.graph.nodes:
            if node.op in ('placeholder', 'output'):
                continue

            if not self._is_supported(node):
                if current:
                    subgraphs.append(ForestTraverser(current).get_schedule())
                    current = []
                    current_shape = None
                continue

            # Reduction ops get their own subgraph (standalone).
            op_desc = get_op_descriptor(node)
            if op_desc is not None and op_desc.kind == OpKind.REDUCTION:
                if current:
                    subgraphs.append(ForestTraverser(current).get_schedule())
                    current = []
                    current_shape = None
                subgraphs.append([node])
                continue

            shape = self._get_shape(node)
            if shape is None:
                if current:
                    subgraphs.append(ForestTraverser(current).get_schedule())
                    current = []
                    current_shape = None
                continue

            # Slice ops change shape and base pointer (for non-zero start).  They
            # must be standalone so the downstream subgraph receives the correctly-
            # offset tensor via PyTorch's view mechanism.
            is_slice = self._is_slice_node(node) if op_desc is not None else False

            if current_shape is None:
                can_add = True
            elif is_slice:
                can_add = False
            else:
                merged = _broadcast_shapes(current_shape, shape)
                can_add = merged is not None

            if can_add:
                candidate = current + [node]
                scheduled = ForestTraverser(candidate).get_schedule()
                if len(self._required_globals(scheduled)) > self.max_tensors:
                    can_add = False
                elif not self._stack_fits(scheduled):
                    can_add = False
                else:
                    merged_shape = _broadcast_shapes(current_shape, shape) if current_shape else shape
                    global_nodes_list = list(self._required_globals(scheduled))
                    global_shapes_raw = [self._get_shape(n) for n in global_nodes_list]
                    global_shapes_raw = self._effective_global_shapes(
                        scheduled, global_nodes_list, global_shapes_raw)
                    global_shapes = [s for s in global_shapes_raw if s is not None]
                    if global_shapes and any(s != merged_shape for s in global_shapes):
                        global_strides = [self._get_strides(n) for n in global_nodes_list]
                        global_strides = [global_strides[i] for i, s in enumerate(global_shapes_raw) if s is not None]
                        if _compute_broadcast_plan(merged_shape, global_shapes, global_strides) is None:
                            can_add = False

            if can_add:
                current_shape = _broadcast_shapes(current_shape, shape) if current_shape else shape
                current.append(node)
            else:
                # Flush whatever valid subgraph we already accumulated; the
                # rejected node forces a new one (or eager fallback).
                if current:
                    subgraphs.append(ForestTraverser(current).get_schedule())
                # Check if the rejected node can start a new subgraph on its own
                # (its globals must be broadcast-compatible with its shape).
                solo = ForestTraverser([node]).get_schedule()
                solo_global_nodes = list(self._required_globals(solo))
                solo_globals_raw = [self._get_shape(n) for n in solo_global_nodes]
                solo_globals_raw = self._effective_global_shapes(
                    solo, solo_global_nodes, solo_globals_raw)
                solo_globals = [s for s in solo_globals_raw if s is not None]
                if solo_globals and any(s != shape for s in solo_globals):
                    solo_strides = [self._get_strides(n) for n in solo_global_nodes]
                    solo_strides = [solo_strides[i] for i, s in enumerate(solo_globals_raw) if s is not None]
                    if _compute_broadcast_plan(shape, solo_globals, solo_strides) is None:
                        # Node can't form a valid subgraph — skip it (run eagerly)
                        current = []
                        current_shape = None
                        continue
                current = [node]
                current_shape = shape

        if current:
            subgraphs.append(ForestTraverser(current).get_schedule())
        return subgraphs

    def _ext_io(self, nodes: List[Node]) -> Set[Node]:
        """External IO of a subgraph: inputs from outside + values consumed
        outside the subgraph (graph outputs or subgraph-external users).

        These always need a global-tensor slot. Multi-user *internal*
        intermediates are handled by the register-spill planner instead;
        the planner returns an ``overflow`` set when its registers run
        out, and the caller adds those to the global set."""
        node_set = set(nodes)
        required: Set[Node] = set()
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, Node) and arg not in node_set:
                    required.add(arg)
            if node in self.graph_outputs or any(u not in node_set for u in node.users):
                required.add(node)
        return required

    def _required_globals(self, nodes: List[Node]) -> Set[Node]:
        """Globals needed for a subgraph: external IO plus the spill-planner
        overflow (multi-user intermediates that didn't fit in any register)."""
        ext_io = self._ext_io(nodes)
        _, overflow, _ = _plan_spills(nodes, ext_io)
        return ext_io | overflow

    def _stack_fits(self, nodes: List[Node]) -> bool:
        """Check that the schedule fits within the l12 hardware budget
        (stack ≤ 8, stack + active_regs ≤ 11) once register spilling is
        applied. The spill planner is strictly more permissive than the
        old depth-only check, so feasible=True iff a legal placement
        exists."""
        ext_io = self._ext_io(nodes)
        _, _, feasible = _plan_spills(nodes, ext_io)
        return feasible

    def _stack_fits_legacy(self, nodes: List[Node]) -> bool:
        """Original depth-only feasibility (kept for reference / debugging).
        Counts every live value as a stack slot — multi-user values are
        treated as living on the stack via the dup+store-to-global pattern."""
        node_set = set(nodes)
        depth = 0
        uses_left: Dict[Node, int] = {}
        for node in nodes:
            uses_left[node] = sum(1 for u in node.users if u in node_set)

        for node in nodes:
            op_desc = get_op_descriptor(node)
            arity = op_desc.arity if op_desc else 0
            peak_extra = op_desc.peak_stack_extra if op_desc else 0
            # Each arg may push one item onto the stack before the op runs.
            depth += arity
            # Peak hardware depth during the op's own emission.
            if depth + peak_extra > self.max_stack:
                return False
            depth -= arity   # op consumes its inputs
            depth += 1       # op pushes its result
            is_ext_out = node in self.graph_outputs or any(u not in node_set for u in node.users)
            if is_ext_out or uses_left.get(node, 0) == 0:
                depth -= 1   # result stored or discarded
            if depth > self.max_stack:
                return False
        return True


# ---------------------------------------------------------------------------
# Main compiler
# ---------------------------------------------------------------------------

class GintCompiler:
    """Converts a functionalized FX graph to one or more gint programs."""

    def __init__(self, gm: GraphModule, example_inputs: List[torch.Tensor]):
        self.gm = gm
        self.example_inputs = example_inputs

    def compile(self) -> Callable:
        partitioner = GraphPartitioner(self.gm, max_tensors=8, max_stack=8)
        raw_schedules = partitioner.partition()

        # Fuse pointwise + reduction + pointwise triples where possible.
        merged_schedules: List[List[Node]] = []
        compiled: Dict[int, GintCompiledSubgraph] = {}
        i = 0
        sg_index = 0
        pending_pw: Optional[List[Node]] = None  # deferred pointwise schedule

        def _emit_pending():
            """Compile and record the deferred pointwise subgraph (if any)."""
            nonlocal sg_index
            if pending_pw is not None:
                compiled[sg_index] = self._compile_subgraph(
                    pending_pw, partitioner)
                merged_schedules.append(pending_pw)
                sg_index += 1

        while i < len(raw_schedules):
            schedule = raw_schedules[i]
            if self._is_reduction_subgraph(schedule):
                reduction_node = schedule[0]
                keepdim = (reduction_node.args[2]
                           if len(reduction_node.args) > 2
                           else reduction_node.kwargs.get('keepdim', False))

                # Check for pre-reduction fusion (deferred pointwise before reduction).
                pre_prefix = None
                if pending_pw is not None:
                    if self._can_fuse_pre_reduction(
                            pending_pw, reduction_node, partitioner):
                        pre_prefix = pending_pw
                        pending_pw = None  # consumed, don't compile
                    else:
                        _emit_pending()

                # Check for post-reduction fusion (existing).
                pw_schedule = None
                if (keepdim and i + 1 < len(raw_schedules)
                        and not self._is_reduction_subgraph(raw_schedules[i + 1])
                        and self._can_fuse_with_reduction(
                            reduction_node, raw_schedules[i + 1], partitioner)):
                    pw_schedule = raw_schedules[i + 1]

                if pre_prefix is not None or pw_schedule is not None:
                    fused_nodes = (pre_prefix or []) + [reduction_node] + (pw_schedule or [])
                    compiled[sg_index] = self._compile_fused_reduction_subgraph(
                        reduction_node, pw_schedule or [], partitioner,
                        pre_prefix=pre_prefix)
                    merged_schedules.append(fused_nodes)
                    sg_index += 1
                    i += 2 if pw_schedule else 1
                    continue
                compiled[sg_index] = self._compile_reduction_subgraph(
                    reduction_node, partitioner)
            else:
                _emit_pending()
                pending_pw = schedule  # defer compilation
                i += 1
                continue
            merged_schedules.append(schedule)
            sg_index += 1
            i += 1

        _emit_pending()

        subgraph_schedules = merged_schedules

        # Map each node → its subgraph index.
        node_to_sg: Dict[Node, int] = {}
        for sg_id, schedule in enumerate(subgraph_schedules):
            for node in schedule:
                node_to_sg[node] = sg_id

        def execute(*args):
            results: Dict[Node, torch.Tensor] = {}

            ph_idx = 0
            for node in self.gm.graph.nodes:
                if node.op == 'placeholder':
                    results[node] = args[ph_idx]
                    ph_idx += 1

            executed_sgs: Set[int] = set()
            for node in self.gm.graph.nodes:
                if node.op in ('placeholder', 'output'):
                    continue
                if node in node_to_sg:
                    sg_id = node_to_sg[node]
                    if sg_id not in executed_sgs:
                        self._run_subgraph(
                            sg_id, subgraph_schedules[sg_id],
                            compiled[sg_id], results, partitioner,
                        )
                        executed_sgs.add(sg_id)
                else:
                    results[node] = self._run_eager(node, results)

            for node in self.gm.graph.nodes:
                if node.op == 'output':
                    res = node.args[0]
                    if isinstance(res, (list, tuple)):
                        return tuple(results[n] if isinstance(n, Node) else n for n in res)
                    return results[res] if isinstance(res, Node) else res

        return execute

    @staticmethod
    def _can_fuse_with_reduction(reduction_node: Node, pw_schedule: List[Node],
                                  partitioner: GraphPartitioner) -> bool:
        """Check if a pointwise schedule can fuse with a preceding keepdim reduction.

        Supports multi-op pointwise chains that split into a scalar prefix (ops
        consuming only the reduction scalar) and a broadcast suffix (ops consuming
        external global tensors that share the reduction dimension).
        """
        pw_set = set(pw_schedule)

        # 1. All nodes must be ELEMENTWISE.
        for node in pw_schedule:
            desc = get_op_descriptor(node)
            if desc is None or desc.kind != OpKind.ELEMENTWISE:
                return False

        # 2. Reduction result must have no users outside the fused subgraph.
        for user in reduction_node.users:
            if user not in pw_set:
                return False

        # 3. Compute the split; broadcast suffix must be non-empty.
        scalar_prefix, broadcast_suffix = GintCompiler._split_pointwise_chain(
            reduction_node, pw_schedule)
        if not broadcast_suffix:
            return False

        # 3b. Every node in pw_schedule must transitively depend on the
        # reduction. Without this, an unrelated pointwise op that happens
        # to share the reduction dim's last-dim size (e.g. the *next*
        # dot product's pre-reduction `n*L` when reduction_size=3) gets
        # pulled into broadcast_suffix; the fused kernel then drops the
        # scalar_prefix output that downstream subgraphs depend on.
        reachable = {reduction_node}
        changed = True
        while changed:
            changed = False
            for node in pw_schedule:
                if node in reachable:
                    continue
                for arg in node.args:
                    if isinstance(arg, Node) and arg in reachable:
                        reachable.add(node)
                        changed = True
                        break
        for node in pw_schedule:
            if node not in reachable:
                return False

        # 4. All external global inputs to broadcast_suffix must share the
        #    reduction dimension.
        reduction_input = reduction_node.args[0]
        input_shape = partitioner._get_shape(reduction_input)
        if input_shape is None:
            return False
        reduction_size = input_shape[-1]

        for node in broadcast_suffix:
            for arg in node.args:
                if not isinstance(arg, Node):
                    continue
                if arg in pw_set or arg is reduction_node:
                    continue
                shape = partitioner._get_shape(arg)
                if shape is None:
                    return False
                if len(shape) == 0:
                    continue  # 0-d scalar tensor, will broadcast
                if shape[-1] != reduction_size:
                    return False

        return True

    @staticmethod
    def _can_fuse_pre_reduction(pre_schedule: List[Node], reduction_node: Node,
                                 partitioner: GraphPartitioner) -> bool:
        """Check if a pointwise subgraph can fuse BEFORE a reduction.

        The pre-prefix ops are emitted inside the Phase 1 accumulation loop:
        each chunk is loaded, the prefix ops are applied, and the result is
        accumulated.  This avoids storing the prefix result to global memory.
        """
        pre_set = set(pre_schedule)

        # 1. All nodes must be ELEMENTWISE.
        for node in pre_schedule:
            desc = get_op_descriptor(node)
            if desc is None or desc.kind != OpKind.ELEMENTWISE:
                return False

        # 2. The reduction input must be a node in the pre-prefix schedule,
        #    and its only consumer must be the reduction.
        reduction_input = reduction_node.args[0]
        if reduction_input not in pre_set:
            return False
        for user in reduction_input.users:
            if user is not reduction_node:
                return False

        # 3. All external inputs to the pre-prefix must share the reduction dim.
        input_shape = partitioner._get_shape(reduction_input)
        if input_shape is None:
            return False
        reduction_size = input_shape[-1]

        for node in pre_schedule:
            for arg in node.args:
                if not isinstance(arg, Node):
                    continue
                if arg in pre_set:
                    continue
                shape = partitioner._get_shape(arg)
                if shape is None:
                    return False
                if len(shape) == 0:
                    continue
                if shape[-1] != reduction_size:
                    return False

        return True

    @staticmethod
    def _split_pointwise_chain(reduction_node: Node, pw_schedule: List[Node]) -> tuple:
        """Split pw_schedule into (scalar_prefix, broadcast_suffix).

        A node belongs to broadcast_suffix if it transitively depends on any
        external global tensor (not in pw_schedule and not the reduction_node).
        """
        pw_set = set(pw_schedule)
        broadcast_set: Set[Node] = set()
        for node in pw_schedule:
            for arg in node.args:
                if not isinstance(arg, Node):
                    continue
                if arg not in pw_set and arg is not reduction_node:
                    broadcast_set.add(node)
                    break
                if arg in broadcast_set:
                    broadcast_set.add(node)
                    break
        scalar_prefix = [n for n in pw_schedule if n not in broadcast_set]
        broadcast_suffix = [n for n in pw_schedule if n in broadcast_set]
        return scalar_prefix, broadcast_suffix

    @staticmethod
    def _vstack_depth(vstack: List[Optional[Node]], node: Node) -> int:
        """Return depth of *node* from top of *vstack* (0 = top), or -1 if absent."""
        for i, n in enumerate(reversed(vstack)):
            if n is node:
                return i
        return -1

    # --- Subgraph compilation ---

    def _compile_subgraph(
        self, nodes: List[Node], partitioner: GraphPartitioner
    ) -> GintCompiledSubgraph:
        # Compute the spill plan first: multi-user intermediates go to
        # virtual registers when possible, with the remaining "overflow"
        # falling back to a global tensor slot exactly like before.
        ext_io = partitioner._ext_io(nodes)
        node_to_reg, overflow, _ = _plan_spills(nodes, ext_io)
        global_nodes = self._sorted_globals(ext_io | overflow)
        tensor_map = {n: i for i, n in enumerate(global_nodes)}

        # Compute broadcast plan FIRST so the tile decision can drive the
        # codegen's load/store frontend choice.
        # Use effective shapes: for slice inputs, the effective shape is the
        # slice OUTPUT shape (the subgraph only accesses the sliced window).
        all_shapes = [partitioner._get_shape(n) for n in global_nodes]
        all_shapes = list(GraphPartitioner._effective_global_shapes(
            nodes, global_nodes, all_shapes))
        all_strides = [partitioner._get_strides(n) for n in global_nodes]

        output_shape = all_shapes[0]
        for s in all_shapes[1:]:
            if s is not None:
                output_shape = _broadcast_shapes(output_shape, s)

        valid_shapes = [s for s in all_shapes if s is not None]
        valid_strides = [all_strides[i] for i, s in enumerate(all_shapes) if s is not None]
        plan = _compute_broadcast_plan(output_shape, valid_shapes, valid_strides)

        # Tile dispatch: when the merge couldn't fully collapse to a flat
        # 1D block (block_size < 128 with at least one batch dim left),
        # switch to a multi-batch-per-warp tile so the warp's 128 lanes
        # are used for parallel batches instead of mostly idle. Mirrors
        # the reduction tile selector exactly.
        block_size = plan['block_size']
        batch_dims = list(plan['batch_dims'])
        if block_size < 128 and len(batch_dims) >= 1:
            tile = _select_pointwise_tiling(block_size)
        else:
            tile = dict(B=1, R=128, mode='1d')
        mode = tile['mode']

        if mode == '1d':
            load_fn, store_fn = fe.fldg_1d, fe.fstg_1d
        elif mode == '2dt':
            load_fn, store_fn = fe.fldg_2dt, fe.fstg_2dt
        else:  # '2dw'
            load_fn, store_fn = fe.fldg_2dw, fe.fstg_2dw

        # Open a FrontendState context so all fe.* calls accumulate into fe_state.bc.
        fe_state = FrontendState([])
        token = _frontend_state.set(fe_state)
        try:
            codegen = StackCodegen(max_stack=8, tensor_map=tensor_map,
                                   load_fn=load_fn, store_fn=store_fn,
                                   node_to_reg=node_to_reg)
            codegen.set_subgraph_nodes(nodes, partitioner.graph_outputs)

            for node in nodes:
                op_desc = get_op_descriptor(node)
                if op_desc is None:
                    raise RuntimeError(f"Unsupported op reached compiler: {node.target}")

                # Handle operands in the prescribed arg order.
                arg_indices = resolve_arg_order(op_desc, node)
                pushed = 0
                for idx in arg_indices:
                    arg = node.args[idx]
                    if isinstance(arg, (Node, int, float)):
                        codegen.handle_operand(arg)
                        pushed += 1

                # Emit the operation and manage the result.
                codegen.emit_op(node, op_desc, num_pushed=pushed)

            fe.halt()
            bytecode = fe_state.bc
        finally:
            _frontend_state.reset(token)

        # Build per-tensor TensorInfos. Layout differs by tile mode:
        #   1d:  block in shape_1 (thread+width flatten), batches in batch_shape
        #   2dt: inner in shape_1 (thread), innermost batch in shape_2 (width)
        #   2dw: innermost batch in shape_1 (thread), inner in shape_2 (width)
        # For 2dt/2dw the innermost merged batch dim is pulled into block_grid
        # so the kernel's `b_shape -= b_idx*step` clamp masks partial last
        # tiles when the batch count isn't a multiple of B.
        tensor_infos = []
        if mode == '1d':
            num_inner_blocks = plan['num_inner_blocks']
            for i, node in enumerate(global_nodes):
                entry = plan['per_tensor'][i]
                tensor_infos.append(ProgramTensorInfo(
                    elm_size=4,
                    batch_strides=entry['batch_strides'],
                    batch_shape=batch_dims,
                    block_shape_stride_1=[block_size, entry['block_stride']],
                    block_shape_stride_2=[1, 0],
                    block_grid_dims=[num_inner_blocks, 1],
                    block_grid_steps=[128, 1],
                ))
            grid_dim = plan['grid_dim']
        else:
            B, R = tile['B'], tile['R']
            innermost_batch = batch_dims[-1]
            outer_batch_dims = batch_dims[:-1]
            inner_chunks = (block_size + R - 1) // R
            batch_tiles = (innermost_batch + B - 1) // B
            for i, node in enumerate(global_nodes):
                entry = plan['per_tensor'][i]
                inner_stride = entry['block_stride']
                bsts = list(entry['batch_strides'])
                inner_batch_stride = bsts[-1]
                outer_batch_strides = bsts[:-1]
                if mode == '2dt':
                    bsst1 = [block_size, inner_stride]
                    bsst2 = [innermost_batch, inner_batch_stride]
                    grid_dims = [inner_chunks, batch_tiles]
                    grid_steps = [R, B]
                else:  # '2dw'
                    bsst1 = [innermost_batch, inner_batch_stride]
                    bsst2 = [block_size, inner_stride]
                    grid_dims = [batch_tiles, inner_chunks]
                    grid_steps = [B, R]
                tensor_infos.append(ProgramTensorInfo(
                    elm_size=4,
                    batch_strides=outer_batch_strides,
                    batch_shape=outer_batch_dims,
                    block_shape_stride_1=bsst1,
                    block_shape_stride_2=bsst2,
                    block_grid_dims=grid_dims,
                    block_grid_steps=grid_steps,
                ))
            grid_dim = inner_chunks * batch_tiles
            for d in outer_batch_dims:
                grid_dim *= d

        # Compute input adjustments for slice ops: non-zero start offsets
        # require adjusting the input tensor's base pointer at runtime.
        input_adjustments = []
        for node in nodes:
            if not GraphPartitioner._is_slice_node(node):
                continue
            inp = node.args[0]
            if not isinstance(inp, Node) or inp not in global_nodes:
                continue
            dim = node.args[1]
            start = node.args[2]
            end = node.args[3]
            step = node.args[4] if len(node.args) > 4 else 1
            if step != 1:
                continue  # stepped slices not supported
            if start == 0:
                continue  # no offset needed
            gidx = global_nodes.index(inp)
            input_adjustments.append((gidx, dim, start))

        return GintCompiledSubgraph(bytecode, tensor_infos,
                                     output_shape=output_shape,
                                     grid_dim=grid_dim,
                                     input_adjustments=input_adjustments)

    # --- Reduction detection and compilation ---

    @staticmethod
    def _is_reduction_subgraph(schedule: List[Node]) -> bool:
        if len(schedule) != 1:
            return False
        desc = get_op_descriptor(schedule[0])
        return desc is not None and desc.kind == OpKind.REDUCTION

    def _compile_reduction_subgraph(
        self, node: Node, partitioner: GraphPartitioner
    ) -> GintCompiledSubgraph:
        """Compile a single innermost-dim reduction.

        Tile decomposition selects (B batches, R reduction-elts) per warp such
        that B * R == 128. Three modes (see _select_reduction_tiling):

          mode '1d'  (B=1,  R=128): threads+width flatten over reduction dim.
                                    Final = warp_allreduce + width-combine.
          mode '2dt' (B=4,  R=32):  threads scan reduction, width-lanes carry
                                    4 batches. Final = warp_allreduce only.
          mode '2dw' (B=32, R=4):   threads carry 32 batches, width-lanes scan
                                    reduction. Final = width-combine only.

        Outer batch dims are merged via _merge_dims; the innermost merged batch
        dim is placed in block_grid (dim_2 for 1d/2dt, dim_1 for 2dw) so the
        kernel's per-warp shape clamping handles partial last tiles when the
        batch count isn't a multiple of B.
        """
        input_node = node.args[0]
        input_shape = list(partitioner._get_shape(input_node))
        output_shape = list(partitioner._get_shape(node))

        N = input_shape[-1]
        keepdim = (node.args[2] if len(node.args) > 2
                   else node.kwargs.get('keepdim', False))

        in_strides = list(partitioner._get_strides(input_node) or
                          _c_contiguous_strides(input_shape))
        out_strides_full = list(partitioner._get_strides(node) or
                                _c_contiguous_strides(output_shape))

        red_stride = in_strides[-1] if in_strides else 1

        batch_shape_raw = list(input_shape[:-1])
        in_batch_strides = list(in_strides[:-1]) if in_strides else []
        if keepdim:
            out_batch_strides = (list(out_strides_full[:-1])
                                 if out_strides_full else [])
        else:
            out_batch_strides = list(out_strides_full)

        merged_shape, merged_strides = _merge_dims(
            batch_shape_raw, [in_batch_strides, out_batch_strides])
        merged_in_bs, merged_out_bs = merged_strides

        tile = _select_reduction_tiling(N)
        B, R, mode = tile['B'], tile['R'], tile['mode']
        n_chunks = math.ceil(N / R)

        global_nodes = self._sorted_globals(partitioner._required_globals([node]))
        tensor_map = {n: i for i, n in enumerate(global_nodes)}
        input_slot = tensor_map[input_node]
        output_slot = tensor_map[node]

        op_desc = get_op_descriptor(node)
        combine_fn = op_desc.combine_fn

        if mode == '1d':
            load_fn, store_fn = fe.fldg_1d, fe.fstg_1d
        elif mode == '2dt':
            load_fn, store_fn = fe.fldg_2dt, fe.fstg_2dt
        else:  # '2dw' input; 2dt store (only width-lane 0 writes)
            load_fn, store_fn = fe.fldg_2dw, fe.fstg_2dt

        fe_state = FrontendState([])
        token = _frontend_state.set(fe_state)
        try:
            # Phase 1: load + accumulate all reduction chunks.
            load_fn(0, input_slot)
            for k in range(1, n_chunks):
                load_fn(k * R, input_slot)
                combine_fn()

            # Phase 2: warp_allreduce / width-combine subset per tile.
            if tile['do_warp_reduce'] and op_desc.warp_reduce_fn is not None:
                op_desc.warp_reduce_fn()
            if tile['do_width_combine']:
                _emit_width_combine(combine_fn)

            # Phase 2b: post step (e.g. mean's *(1/N)).
            if op_desc.post_reduce_fn is not None:
                op_desc.post_reduce_fn(node)

            # Phase 3: store.
            store_fn(0, output_slot)
            fe.halt()
            bytecode = fe_state.bc
        finally:
            _frontend_state.reset(token)

        # Innermost merged batch dim drives the per-warp B-tile; outer dims
        # stay in batch_shape and iterate via the kernel's batch decomposition.
        if not merged_shape:
            m_inner = 1
            in_inner_stride = 1
            out_inner_stride = 1
            outer_shape: tuple = ()
            outer_in_bs: tuple = ()
            outer_out_bs: tuple = ()
        else:
            m_inner = merged_shape[-1]
            in_inner_stride = merged_in_bs[-1] if merged_in_bs else 1
            out_inner_stride = merged_out_bs[-1] if merged_out_bs else 1
            outer_shape = merged_shape[:-1]
            outer_in_bs = merged_in_bs[:-1] if merged_in_bs else ()
            outer_out_bs = merged_out_bs[:-1] if merged_out_bs else ()

        tensor_infos = []
        for gn in global_nodes:
            is_input = gn is input_node
            bs_inner = in_inner_stride if is_input else out_inner_stride
            bs_outer = list(outer_in_bs) if is_input else list(outer_out_bs)

            if mode == '1d':
                if is_input:
                    bsst1 = [N, red_stride]
                    bsst2 = [m_inner, bs_inner]
                else:
                    bsst1 = [1, 1]
                    bsst2 = [m_inner, bs_inner]
                grid_dim_1, grid_dim_2 = 1, m_inner
                grid_step_1 = 128 if is_input else 1
                grid_step_2 = 1
            elif mode == '2dt':
                if is_input:
                    bsst1 = [N, red_stride]
                    bsst2 = [m_inner, bs_inner]
                else:
                    bsst1 = [1, 0]
                    bsst2 = [m_inner, bs_inner]
                grid_dim_1 = 1
                grid_dim_2 = (m_inner + B - 1) // B
                grid_step_1 = 32 if is_input else 1
                grid_step_2 = B
            else:  # '2dw'
                if is_input:
                    bsst1 = [m_inner, bs_inner]
                    bsst2 = [N, red_stride]
                else:
                    bsst1 = [m_inner, bs_inner]
                    bsst2 = [1, 0]
                grid_dim_1 = (m_inner + B - 1) // B
                grid_dim_2 = 1
                grid_step_1 = B
                grid_step_2 = 4 if is_input else 1

            tensor_infos.append(ProgramTensorInfo(
                elm_size=4,
                batch_strides=list(bs_outer),
                batch_shape=list(outer_shape),
                block_shape_stride_1=bsst1,
                block_shape_stride_2=bsst2,
                block_grid_dims=[grid_dim_1, grid_dim_2],
                block_grid_steps=[grid_step_1, grid_step_2],
            ))

        grid_dim = grid_dim_1 * grid_dim_2
        for d in outer_shape:
            grid_dim *= d
        grid_dim = max(1, grid_dim)

        return GintCompiledSubgraph(
            bytecode, tensor_infos,
            output_shape=tuple(output_shape) if output_shape else (),
            grid_dim=grid_dim,
        )

    def _compile_fused_reduction_new_tile(
        self, reduction_node: Node, pw_schedule: List[Node],
        partitioner: GraphPartitioner,
        *,
        pre_prefix: Optional[List[Node]],
        scalar_prefix: List[Node],
        broadcast_suffix: List[Node],
        tile: dict,
    ) -> GintCompiledSubgraph:
        """Tile-aware fused reduction (modes '2dt' and '2dw').

        Handles all three phase combinations:
          - pre_prefix: pointwise ops absorbed into Phase 1's accumulation
          - scalar_prefix: per-thread pointwise ops on the reduced scalar
          - broadcast_suffix: per-chunk pointwise ops mixing the scalar with
            external (..., N)-shaped globals; each chunk emits load + ops +
            store using the tile's block-shaped layout

        Mode '1d' (large N) stays on the legacy `_compile_fused_reduction_subgraph`
        codegen below — that path is structurally different (batches in
        `batch_shape` rather than `block_grid`) and well-tested by RMSNorm.
        """
        pre_prefix = list(pre_prefix or [])
        scalar_prefix = list(scalar_prefix)
        broadcast_suffix = list(broadcast_suffix)

        reduction_input = reduction_node.args[0]
        input_shape = list(partitioner._get_shape(reduction_input))
        N = input_shape[-1]
        keepdim = (reduction_node.args[2] if len(reduction_node.args) > 2
                   else reduction_node.kwargs.get('keepdim', False))

        if broadcast_suffix:
            output_node = broadcast_suffix[-1]
        elif scalar_prefix:
            output_node = scalar_prefix[-1]
        else:
            output_node = reduction_node
        output_shape = list(partitioner._get_shape(output_node))

        all_nodes = pre_prefix + [reduction_node] + scalar_prefix + broadcast_suffix
        node_set = set(all_nodes)
        global_nodes = self._sorted_globals(partitioner._required_globals(all_nodes))
        tensor_map = {n: i for i, n in enumerate(global_nodes)}

        # Phase 2 stores: broadcast_suffix nodes that have external users.
        suffix_output_nodes: Set[Node] = {
            n for n in broadcast_suffix
            if n in tensor_map and (
                n in partitioner.graph_outputs
                or any(u not in node_set for u in n.users))
        }

        # Globals split into "scalar" (single-element-per-batch — only the
        # reduction-only output when broadcast_suffix is empty) and "block"
        # (full reduction-dim block per warp tile — everything else).
        scalar_globals: Set[Node] = set()
        if not broadcast_suffix and output_node in tensor_map:
            scalar_globals.add(output_node)

        # Per-tensor batch strides for the subgraph's batch axis
        # (input_shape[:-1]). Pads with 0-strides on the left for tensors
        # with fewer batch dims (broadcast over outer batches), and uses
        # 0 within-shape for size-1 batch dims (broadcast within the same
        # ndim).
        batch_shape_raw = list(input_shape[:-1])

        def _is_block_global(gn: Node) -> bool:
            return gn not in scalar_globals

        def _batch_strides_for(gn: Node) -> List[int]:
            gn_shape = partitioner._get_shape(gn)
            gn_strides = partitioner._get_strides(gn)
            if gn_shape is None:
                gn_shape = ()
            gn_shape = list(gn_shape)
            if _is_block_global(gn):
                # Block-shaped: tensor's shape ends in the reduction dim
                # (size N or 1 for inner-broadcast). Batch portion is the
                # leading dims.
                actual_shape = gn_shape[:-1] if gn_shape else []
                if gn_strides is None:
                    actual_strides = _c_contiguous_strides(gn_shape)[:-1] if gn_shape else []
                else:
                    actual_strides = list(gn_strides[:-1]) if gn_strides else []
            else:
                # Scalar-output: actual batch portion is full shape (no
                # reduction dim) for keepdim=False, or shape[:-1] for
                # keepdim=True.
                if keepdim and gn_shape:
                    actual_shape = gn_shape[:-1]
                    if gn_strides is None:
                        actual_strides = _c_contiguous_strides(gn_shape)[:-1]
                    else:
                        actual_strides = list(gn_strides[:-1])
                else:
                    actual_shape = list(gn_shape)
                    if gn_strides is None:
                        actual_strides = _c_contiguous_strides(gn_shape)
                    else:
                        actual_strides = list(gn_strides)
            n_pad = len(batch_shape_raw) - len(actual_shape)
            padded = [0] * max(0, n_pad)
            # Within the tensor's existing dims, broadcast (size 1 vs
            # subgraph dim > 1) → 0 stride.
            for i, sz in enumerate(actual_shape):
                out_dim = batch_shape_raw[max(0, n_pad) + i]
                if sz == 1 and out_dim > 1:
                    padded.append(0)
                else:
                    padded.append(actual_strides[i] if i < len(actual_strides) else 0)
            return padded

        per_tensor_batch_strides_full: Dict[Node, List[int]] = {
            gn: _batch_strides_for(gn) for gn in global_nodes
        }

        def _inner_stride_for(gn: Node) -> int:
            if gn in scalar_globals:
                return 0  # unused (bsst has shape=1 for the scalar dim)
            gn_shape = partitioner._get_shape(gn)
            if gn_shape is None or len(gn_shape) == 0:
                return 0
            if gn_shape[-1] == 1 and N > 1:
                return 0  # broadcast in inner
            gn_strides = partitioner._get_strides(gn)
            if gn_strides is not None and len(gn_strides) > 0:
                return gn_strides[-1]
            return 1  # C-contig fallback

        per_tensor_inner_stride: Dict[Node, int] = {
            gn: _inner_stride_for(gn) for gn in global_nodes
        }

        # Joint merge: every tensor in the subgraph must agree to merge a
        # batch-dim pair. Tensors broadcasting outer dims (stride 0) merge
        # with each other; the merge stops at the first dim where some
        # tensor doesn't fit `stride[i] == stride[i+1] * shape[i+1]`.
        all_strides_lists = [per_tensor_batch_strides_full[gn] for gn in global_nodes]
        merged_batch_shape, merged_strides_per_tensor = _merge_dims(
            batch_shape_raw, all_strides_lists)

        if not merged_batch_shape:
            m_inner = 1
            outer_shape: tuple = ()
            per_tensor_inner_batch_stride = {gn: 0 for gn in global_nodes}
            per_tensor_outer_strides = {gn: () for gn in global_nodes}
        else:
            m_inner = merged_batch_shape[-1]
            outer_shape = merged_batch_shape[:-1]
            per_tensor_inner_batch_stride = {}
            per_tensor_outer_strides = {}
            for i, gn in enumerate(global_nodes):
                ms = merged_strides_per_tensor[i]
                per_tensor_inner_batch_stride[gn] = ms[-1] if ms else 0
                per_tensor_outer_strides[gn] = tuple(ms[:-1]) if ms else ()

        B, R, mode = tile['B'], tile['R'], tile['mode']
        n_chunks = math.ceil(N / R)

        if mode == '2dt':
            block_load_fn = fe.fldg_2dt
            block_store_fn = fe.fstg_2dt
        else:  # '2dw'
            block_load_fn = fe.fldg_2dw
            block_store_fn = fe.fstg_2dw
        # Scalar-output store always uses 2DT addressing — for both modes,
        # `(t * stride_1) + (w * stride_2)` with shape_2 = 1 / stride_2 = 0
        # masks down to a single (thread, lane) write per batch lane and
        # the addressing is identical between fstg_2dt and fstg_2dw at
        # operand 0. Picking 2dt for both keeps it simple.
        scalar_store_fn = fe.fstg_2dt

        op_desc_red = get_op_descriptor(reduction_node)
        combine_fn = op_desc_red.combine_fn

        fe_state = FrontendState([])
        token = _frontend_state.set(fe_state)
        try:
            # Phase 1: load + accumulate (with optional pre_prefix ops).
            if pre_prefix:
                for k in range(n_chunks):
                    for nd in pre_prefix:
                        desc = get_op_descriptor(nd)
                        arg_order = resolve_arg_order(desc, nd)
                        for idx in arg_order:
                            arg = nd.args[idx]
                            if isinstance(arg, (int, float)):
                                fe.fpush(float(arg))
                            elif isinstance(arg, Node) and arg in tensor_map:
                                block_load_fn(k * R, tensor_map[arg])
                            # else: intermediate Node already on top of stack
                        desc.emit_fn(nd)
                    if k > 0:
                        combine_fn()
            else:
                input_slot = tensor_map[reduction_input]
                block_load_fn(0, input_slot)
                for k in range(1, n_chunks):
                    block_load_fn(k * R, input_slot)
                    combine_fn()

            # Phase 1 final: warp_allreduce / width-combine subset per tile.
            if tile['do_warp_reduce'] and op_desc_red.warp_reduce_fn is not None:
                op_desc_red.warp_reduce_fn()
            if tile['do_width_combine']:
                _emit_width_combine(combine_fn)
            if op_desc_red.post_reduce_fn is not None:
                op_desc_red.post_reduce_fn(reduction_node)

            # Phase 1b: scalar prefix ops on the per-thread (or per-width-lane)
            # scalar. Pointwise ops act per-(thread, width-lane) so the
            # scalar's distribution across the warp doesn't matter.
            for nd in scalar_prefix:
                desc = get_op_descriptor(nd)
                arg_order = resolve_arg_order(desc, nd)
                arg_counts: Dict[Node, int] = {}
                for idx in arg_order:
                    arg = nd.args[idx]
                    if isinstance(arg, Node):
                        arg_counts[arg] = arg_counts.get(arg, 0) + 1
                for idx in arg_order:
                    arg = nd.args[idx]
                    if isinstance(arg, (int, float)):
                        fe.fpush(float(arg))
                    elif isinstance(arg, Node):
                        if arg_counts.get(arg, 0) > 1:
                            fe.dup()
                            arg_counts[arg] -= 1
                desc.emit_fn(nd)

            if broadcast_suffix:
                # Phase 2: per-chunk broadcast suffix.
                scalar_node = scalar_prefix[-1] if scalar_prefix else reduction_node
                for k in range(n_chunks):
                    fe.dup()  # preserve scalar across iterations
                    chunk_vstack: List[Optional[Node]] = [scalar_node]
                    for nd in broadcast_suffix:
                        desc = get_op_descriptor(nd)
                        arg_order = resolve_arg_order(desc, nd)
                        pushed = 0
                        for idx in arg_order:
                            arg = nd.args[idx]
                            if isinstance(arg, (int, float)):
                                fe.fpush(float(arg))
                                chunk_vstack.append(None)
                                pushed += 1
                            elif isinstance(arg, Node):
                                if arg in tensor_map:
                                    block_load_fn(k * R, tensor_map[arg])
                                    chunk_vstack.append(arg)
                                    pushed += 1
                                else:
                                    depth = GintCompiler._vstack_depth(
                                        chunk_vstack, arg)
                                    if depth == -1:
                                        raise RuntimeError(
                                            f"Arg {arg.name} not on chunk "
                                            f"stack for {nd.name}")
                                    if depth == 0:
                                        pass
                                    elif depth == 1:
                                        fe.swap()
                                        top = chunk_vstack.pop()
                                        sec = chunk_vstack.pop()
                                        chunk_vstack.append(top)
                                        chunk_vstack.append(sec)
                                    else:
                                        raise RuntimeError(
                                            f"Arg {arg.name} buried at "
                                            f"depth {depth} for {nd.name}")
                                    pushed += 1
                        desc.emit_fn(nd)
                        for _ in range(pushed):
                            chunk_vstack.pop()
                        chunk_vstack.append(nd)

                        if nd in suffix_output_nodes:
                            block_store_fn(k * R, tensor_map[nd])
                            chunk_vstack.pop()

                    while chunk_vstack:
                        fe.pop()
                        chunk_vstack.pop()

                fe.pop()  # pop the original scalar that survived all chunks
            else:
                if output_node in tensor_map:
                    scalar_store_fn(0, tensor_map[output_node])
                else:
                    fe.pop()
            fe.halt()
            bytecode = fe_state.bc
        finally:
            _frontend_state.reset(token)

        # ---- Build TensorInfos ----
        tensor_infos = []
        for gn in global_nodes:
            is_scalar = gn in scalar_globals
            bs_inner_batch = per_tensor_inner_batch_stride[gn]
            bs_outer = list(per_tensor_outer_strides[gn])
            red_stride_gn = per_tensor_inner_stride[gn]

            if mode == '2dt':
                if not is_scalar:
                    bsst1 = [N, red_stride_gn]
                    bsst2 = [m_inner, bs_inner_batch]
                else:
                    bsst1 = [1, 0]
                    bsst2 = [m_inner, bs_inner_batch]
                grid_dim_1 = 1
                grid_dim_2 = (m_inner + B - 1) // B
                grid_step_1 = 32 if not is_scalar else 1
                grid_step_2 = B
            else:  # '2dw'
                if not is_scalar:
                    bsst1 = [m_inner, bs_inner_batch]
                    bsst2 = [N, red_stride_gn]
                else:
                    bsst1 = [m_inner, bs_inner_batch]
                    bsst2 = [1, 0]
                grid_dim_1 = (m_inner + B - 1) // B
                grid_dim_2 = 1
                grid_step_1 = B
                grid_step_2 = 4 if not is_scalar else 1

            tensor_infos.append(ProgramTensorInfo(
                elm_size=4,
                batch_strides=bs_outer,
                batch_shape=list(outer_shape),
                block_shape_stride_1=bsst1,
                block_shape_stride_2=bsst2,
                block_grid_dims=[grid_dim_1, grid_dim_2],
                block_grid_steps=[grid_step_1, grid_step_2],
            ))

        grid_dim = grid_dim_1 * grid_dim_2
        for d in outer_shape:
            grid_dim *= d
        grid_dim = max(1, grid_dim)

        return GintCompiledSubgraph(
            bytecode, tensor_infos,
            output_shape=tuple(output_shape) if output_shape else (),
            grid_dim=grid_dim,
        )

    def _compile_fused_reduction_subgraph(
        self, reduction_node: Node, pw_schedule: List[Node],
        partitioner: GraphPartitioner,
        pre_prefix: Optional[List[Node]] = None,
    ) -> GintCompiledSubgraph:
        """Compile a fused pointwise+reduction+pointwise subgraph.

        Phases:
        Phase 1:  Load (apply pre-prefix) chunks, accumulate, warp-reduce.
        Phase 1b: Emit scalar prefix ops (consume only the scalar + constants).
        Phase 2:  For each chunk, dup scalar, emit broadcast suffix ops, store.

        When the chosen reduction tile isn't '1d' and there is no broadcast
        suffix (Phase 2 empty), we take a separate small-N path that mirrors
        `_compile_reduction_subgraph`'s tile-aware emission with pre/scalar
        prefix support. The legacy 1d path below handles everything else,
        including the RMSNorm-style broadcast-suffix case.
        """
        scalar_prefix, broadcast_suffix = GintCompiler._split_pointwise_chain(
            reduction_node, pw_schedule)

        reduction_input = reduction_node.args[0]
        input_shape = list(partitioner._get_shape(reduction_input))
        reduction_size = input_shape[-1]

        tile = _select_reduction_tiling(reduction_size)
        if tile['mode'] != '1d':
            return self._compile_fused_reduction_new_tile(
                reduction_node, pw_schedule, partitioner,
                pre_prefix=pre_prefix,
                scalar_prefix=scalar_prefix,
                broadcast_suffix=broadcast_suffix,
                tile=tile,
            )

        n_chunks = math.ceil(reduction_size / 128)

        # Determine the output node: broadcast suffix if present, otherwise
        # the last scalar-prefix node, otherwise the reduction itself.
        if broadcast_suffix:
            output_node = broadcast_suffix[-1]
        elif scalar_prefix:
            output_node = scalar_prefix[-1]
        else:
            output_node = reduction_node
        output_shape = list(partitioner._get_shape(output_node))

        all_nodes = (pre_prefix or []) + [reduction_node] + pw_schedule
        global_nodes = self._sorted_globals(partitioner._required_globals(all_nodes))
        tensor_map = {n: i for i, n in enumerate(global_nodes)}
        node_set = set(all_nodes)
        output_nodes = {n for n in broadcast_suffix
                        if n in tensor_map and (
                           n in partitioner.graph_outputs or
                           any(u not in node_set for u in n.users))}

        fe_state = FrontendState([])
        token = _frontend_state.set(fe_state)
        try:
            red_desc = get_op_descriptor(reduction_node)
            combine_fn = red_desc.combine_fn

            # Phase 1: Load, apply pre-prefix ops, accumulate, warp-reduce
            if pre_prefix:
                # Emit pre-prefix ops inside the accumulation loop: each
                # chunk is loaded, transformed, then accumulated.
                for k in range(n_chunks):
                    for node in pre_prefix:
                        desc = get_op_descriptor(node)
                        arg_order = resolve_arg_order(desc, node)
                        for idx in arg_order:
                            arg = node.args[idx]
                            if isinstance(arg, (int, float)):
                                fe.fpush(float(arg))
                            elif arg in tensor_map:
                                fe.fldg_1d(k * 128, tensor_map[arg])
                        desc.emit_fn(node)
                    if k == 0:
                        pass  # first value starts the accumulator
                    else:
                        combine_fn()
            else:
                input_slot = tensor_map[reduction_input]
                fe.fldg_1d(0, input_slot)
                for k in range(1, n_chunks):
                    fe.fldg_1d(k * 128, input_slot)
                    combine_fn()
            red_desc.emit_fn(reduction_node)
            # Stack now: [scalar_reduction_result]

            # Phase 1b: Emit scalar prefix ops (once, before the chunk loop).
            for node in scalar_prefix:
                desc = get_op_descriptor(node)
                arg_order = resolve_arg_order(desc, node)
                # Count Node-arg appearances to emit dup when the same arg is
                # used multiple times by the same op (e.g. mul(s, s)).
                arg_counts: Dict[Node, int] = {}
                for idx in arg_order:
                    arg = node.args[idx]
                    if isinstance(arg, Node):
                        arg_counts[arg] = arg_counts.get(arg, 0) + 1

                for idx in arg_order:
                    arg = node.args[idx]
                    if isinstance(arg, (int, float)):
                        fe.fpush(float(arg))
                    elif isinstance(arg, Node):
                        if arg_counts.get(arg, 0) > 1:
                            fe.dup()
                            arg_counts[arg] -= 1
                desc.emit_fn(node)
            # Stack now: [scalar_reduction_result_modified]

            if broadcast_suffix:
                # Phase 2: Per-chunk broadcast suffix
                scalar_node = scalar_prefix[-1] if scalar_prefix else reduction_node

                for k in range(n_chunks):
                    fe.dup()  # duplicate scalar for this iteration
                    chunk_vstack: List[Optional[Node]] = [scalar_node]

                    for node in broadcast_suffix:
                        desc = get_op_descriptor(node)
                        arg_order = resolve_arg_order(desc, node)

                        pushed = 0
                        for idx in arg_order:
                            arg = node.args[idx]
                            if isinstance(arg, (int, float)):
                                fe.fpush(float(arg))
                                chunk_vstack.append(None)
                                pushed += 1
                            elif isinstance(arg, Node):
                                if arg in tensor_map:
                                    # Global tensor: load with chunk offset
                                    fe.fldg_1d(k * 128, tensor_map[arg])
                                    chunk_vstack.append(arg)
                                    pushed += 1
                                else:
                                    # Scalar / intermediate: on chunk_vstack
                                    depth = GintCompiler._vstack_depth(chunk_vstack, arg)
                                    if depth == -1:
                                        raise RuntimeError(
                                            f"Arg {arg.name} not on chunk stack "
                                            f"for {node.name}")
                                    if depth == 0:
                                        pass  # already on top
                                    elif depth == 1:
                                        fe.swap()
                                        top = chunk_vstack.pop()
                                        sec = chunk_vstack.pop()
                                        chunk_vstack.append(top)
                                        chunk_vstack.append(sec)
                                    else:
                                        raise RuntimeError(
                                            f"Arg {arg.name} buried at depth {depth} "
                                            f"for {node.name}")
                                    pushed += 1

                        desc.emit_fn(node)
                        for _ in range(pushed):
                            chunk_vstack.pop()
                        chunk_vstack.append(node)

                        if node in output_nodes:
                            fe.fstg_1d(k * 128, tensor_map[node])
                            chunk_vstack.pop()

                    # Pop any leftover intermediates from this iteration
                    # (e.g. scalar if unused by broadcast suffix).
                    while chunk_vstack:
                        fe.pop()
                        chunk_vstack.pop()

                # Pop the original scalar (survived all iterations via dup).
                fe.pop()
            else:
                # No broadcast suffix: store the scalar result (once per batch element).
                if output_node in tensor_map:
                    fe.fstg_1d(0, tensor_map[output_node])
                else:
                    fe.pop()  # consume unused scalar
            fe.halt()
            bytecode = fe_state.bc
        finally:
            _frontend_state.reset(token)

        # Build tensor infos — per-tensor batch strides account for broadcast.
        batch_dims = input_shape[:-1] if len(input_shape) > 1 else []
        # Identify "input" globals (those read from, not written to).
        # Output globals have block size matching their last dim; inputs get
        # the full reduction block.
        input_globals = set()
        for node in (pre_prefix or []):
            input_globals.update(n for n in node.args if isinstance(n, Node) and n in tensor_map)
        input_globals.add(reduction_input)
        if broadcast_suffix:
            for node in broadcast_suffix:
                input_globals.update(n for n in node.args
                                     if isinstance(n, Node) and n in tensor_map)

        tensor_infos = []
        for gn in global_nodes:
            gn_shape = partitioner._get_shape(gn)
            if gn_shape is not None and len(gn_shape) >= 1:
                gn_batch_ndim = len(gn_shape) - 1
                missing = len(batch_dims) - gn_batch_ndim
                gn_bstrides = [0] * max(0, missing)
                if gn_batch_ndim > 0:
                    gn_full_strides = _c_contiguous_strides(gn_shape)
                    gn_bstrides = gn_bstrides + list(gn_full_strides[:-1])
                block_size = gn_shape[-1]
            else:
                gn_bstrides = []
                block_size = 1
            # Input globals use the full reduction block; output globals use
            # their actual block size (1 for scalars, reduction_size for
            # full-size broadcast outputs).
            if gn in input_globals:
                block_size = reduction_size
                grid_steps = [128, 1]
            else:
                grid_steps = [block_size, 1]
            tensor_infos.append(ProgramTensorInfo(
                elm_size=4,
                batch_strides=list(gn_bstrides),
                batch_shape=list(batch_dims),
                block_shape_stride_1=[block_size, 1],
                block_shape_stride_2=[1, 0],
                block_grid_dims=[1, 1],
                block_grid_steps=list(grid_steps),
            ))

        grid_dim = max(1, int(np.prod(batch_dims))) if batch_dims else 1

        return GintCompiledSubgraph(
            bytecode, tensor_infos,
            output_shape=tuple(output_shape),
            grid_dim=grid_dim,
        )

    # --- Subgraph execution ---

    def _run_subgraph(
        self,
        sg_id: int,
        nodes: List[Node],
        compiled: GintCompiledSubgraph,
        results: Dict[Node, torch.Tensor],
        partitioner: GraphPartitioner,
    ):
        node_set = set(nodes)
        global_nodes = self._sorted_globals(partitioner._required_globals(nodes))

        inputs  = [n for n in global_nodes if n not in node_set]
        outputs = [n for n in global_nodes if n in node_set]

        input_tensors  = [results[n] for n in inputs]

        # Apply slice adjustments: narrow input tensors at the slice start offset.
        for gidx, dim, start in compiled.input_adjustments:
            node = global_nodes[gidx]
            if node in inputs:
                pos = inputs.index(node)
                inp_shape = partitioner._get_shape(node)
                if inp_shape is not None:
                    length = inp_shape[dim] - start
                    input_tensors[pos] = input_tensors[pos].narrow(dim, start, length)

        output_tensors = []
        ref = input_tensors[0] if input_tensors else results[next(iter(results))]
        for node in outputs:
            # Each output gets its FX-declared shape, not the subgraph's
            # broadcast iteration shape. Otherwise an (M, 1) output mixed
            # with an (M, 3) output in the same pointwise subgraph would
            # be allocated (M, 3); the kernel's per-tensor TensorInfo
            # already encodes the correct stride-0 broadcast write, but
            # the surplus columns would never be touched and downstream
            # consumers would see garbage in the wrong-shaped tensor.
            out_shape = partitioner._get_shape(node)
            if out_shape is None:
                out_shape = compiled.output_shape
            t = torch.empty(out_shape, dtype=ref.dtype, device=ref.device)
            output_tensors.append(t)
            results[node] = t

        all_tensors = {n: t for n, t in zip(inputs + outputs, input_tensors + output_tensors)}
        ordered     = [all_tensors[n] for n in global_nodes]

        compiled(*ordered, grid_dim=compiled.grid_dim,
                 cuda_stream=torch.cuda.current_stream().cuda_stream)

    def _run_eager(self, node: Node, results: Dict[Node, torch.Tensor]):
        args   = tuple(results[a] if isinstance(a, Node) else a for a in node.args)
        kwargs = {k: results[v] if isinstance(v, Node) else v for k, v in node.kwargs.items()}
        return node.target(*args, **kwargs)

    # --- Utilities ---

    @staticmethod
    def _sorted_globals(nodes: Set[Node]) -> List[Node]:
        return sorted(nodes, key=lambda n: n.name if hasattr(n, 'name') else str(id(n)))

    @staticmethod
    def _numel(global_nodes: List[Node], subgraph_nodes: List[Node]) -> int:
        for node in list(global_nodes) + list(subgraph_nodes):
            if 'val' in node.meta and hasattr(node.meta['val'], 'numel'):
                return node.meta['val'].numel()
            if 'tensor_meta' in node.meta:
                shape = node.meta['tensor_meta'].shape
                n = 1
                for d in shape:
                    n *= d
                return n
        return 1
