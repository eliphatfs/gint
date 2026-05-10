"""Heterogeneous-RL reward benchmark: gint indirect dispatch vs eager serial.

Setup mirrors the typical mock-rollout / replay-evaluation pattern in
multi-task RL: 20 different tasks each have their own dense reward
formula and 128 scenes worth of synthetic state to score.  We do **not**
run physics — the state tensor is filled with random numbers in plausible
ranges; we only compare the cost of dispatching 20 distinct reward
kernels two ways:

* ``eager``: a Python loop that calls each task's torch reward function
  on its slice of the state.  Each call lowers to ~5-15 small CUDA
  kernels (sub, square, sum, sqrt, exp, ...), giving us ~150 launches
  per sweep — exactly the regime where launch overhead dominates.
* ``gint``: a single ``execute_indirect`` launch where 20 warps each
  load their own bytecode and tensor-info pointers from per-warp tables
  and run their reward in the device-side interpreter.  One launch, all
  tasks scored.

We benchmark gint in two flavors:

  * ``execute_indirect`` — the high-level API; retraces every program
    and re-uploads bytecode/tinfo on every call (one-shot convenience);
  * ``PreparedIndirectDispatch.launch()`` — pre-built once, then only
    the bare ``cuLaunchKernel`` runs in the timing loop.  This is what
    a real RL system would use after warmup.

The eager path enjoys torch's own kernel-cache, so comparing it to the
prepared gint launch is the apples-to-apples kernel-time comparison.

Run with:  python -m examples.rl_rewards
"""
import time
import argparse
import torch

from gint.host.executor import get_executor
from .rewards import TASKS, NUM_TASKS, NUM_SCENES, NUM_FEATURES
from .dispatch import PreparedIndirectDispatch


def make_inputs(seed: int = 0):
    """Build the (state, eager_reward, gint_reward) tensors.

    State features are filled with values chosen so the reward arithmetic
    stays in well-conditioned ranges (positions in [-1, 1], scalars
    spanning their plausible domains).  Boolean-style feature slots
    (grasp, placed, success, fall) are sampled as 0/1 floats so
    fselect-style masks are exercised in both branches.
    """
    g = torch.Generator(device='cuda').manual_seed(seed)
    state = torch.empty(NUM_TASKS, NUM_FEATURES, NUM_SCENES,
                        device='cuda', dtype=torch.float32)
    state[:, 0:9].uniform_(-1.0, 1.0, generator=g)
    state[:, 9:16].uniform_(-1.0, 1.0, generator=g)
    bern = (torch.rand(NUM_TASKS, 4, NUM_SCENES,
                       device='cuda', generator=g) > 0.5).float()
    state[:, 10] = bern[:, 0]   # grasp
    state[:, 11] = bern[:, 1]   # placed
    state[:, 12] = bern[:, 2]   # success / fall (task-dependent semantic)
    state[:, 13] = bern[:, 3]   # alt boolean

    eager_reward = torch.zeros(NUM_TASKS, NUM_SCENES,
                               device='cuda', dtype=torch.float32)
    gint_reward = torch.zeros(NUM_TASKS, NUM_SCENES,
                              device='cuda', dtype=torch.float32)
    return state, eager_reward, gint_reward


def run_eager(state, reward):
    """Eager-serial baseline: one torch reward per task, per call."""
    for t, (_, _, torch_fn) in enumerate(TASKS):
        reward[t] = torch_fn(state[t])


def run_gint_indirect(state, reward, executor):
    """Convenience-API gint indirect dispatch.

    Re-traces every program and re-uploads device buffers on each call;
    useful for one-shot use, but its setup cost dominates the kernel
    time.  Use ``PreparedIndirectDispatch`` for hot-loop dispatch.
    """
    programs = [gint_fn for _, gint_fn, _ in TASKS]
    args_list = [(state[t], reward[t]) for t in range(NUM_TASKS)]
    indices = list(range(NUM_TASKS))
    executor.execute_indirect(programs=programs,
                              args_list=args_list,
                              indices=indices)


def time_one(fn, *, warmup: int, repeat: int, label: str):
    """Time *fn* on the default CUDA stream, returning (gpu_us, wall_us) per iter."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    wall_t0 = time.perf_counter()
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - wall_t0) * 1e6 / repeat
    gpu_us = start.elapsed_time(end) * 1000.0 / repeat
    print(f"  {label:38s} gpu={gpu_us:8.1f} us/iter  wall={wall_us:8.1f} us/iter")
    return gpu_us, wall_us


def verify_correctness(state, atol=1e-4, rtol=1e-4):
    """Run both paths on the same inputs and assert per-task agreement."""
    executor = get_executor()
    _, eager_r, gint_r = make_inputs(seed=42)
    state2 = state.clone()
    run_eager(state2, eager_r)
    torch.cuda.synchronize()
    run_gint_indirect(state2, gint_r, executor)
    torch.cuda.synchronize()
    print("Correctness check (eager vs gint, per task):")
    all_ok = True
    for t, (name, _, _) in enumerate(TASKS):
        max_abs = (eager_r[t] - gint_r[t]).abs().max().item()
        ok = torch.allclose(eager_r[t], gint_r[t], atol=atol, rtol=rtol)
        all_ok = all_ok and ok
        status = 'ok ' if ok else 'FAIL'
        print(f"  [{status}] {name:22s} max|Δ|={max_abs:.3e}")
    if not all_ok:
        raise SystemExit("correctness check failed")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the eager-vs-gint correctness check")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    state, eager_reward, gint_reward = make_inputs(args.seed)
    executor = get_executor()

    print(f"Benchmark: {NUM_TASKS} tasks × {NUM_SCENES} scenes × "
          f"{NUM_FEATURES} features (state {tuple(state.shape)}, "
          f"reward {tuple(eager_reward.shape)})")
    print(f"Warp size: {executor.warp_size()};  one warp per task "
          f"({NUM_SCENES} scenes per warp via 32 lanes × REGW=4).")
    print(f"warmup={args.warmup}  repeat={args.repeat}")
    print()

    if not args.skip_verify:
        verify_correctness(state)

    # Pre-build the indirect dispatch (bytecode tracing, device buffer
    # allocations, pointer tables, kernel arg array all done here).
    prepared = PreparedIndirectDispatch(
        executor=executor,
        programs=[gint_fn for _, gint_fn, _ in TASKS],
        args_list=[(state[t], gint_reward[t]) for t in range(NUM_TASKS)],
        indices=list(range(NUM_TASKS)),
    )

    print("Timing (per-iter where one iter = scoring all 20 tasks once):")
    gpu_eager, wall_eager = time_one(
        lambda: run_eager(state, eager_reward),
        warmup=args.warmup, repeat=args.repeat,
        label="eager torch (20 calls/iter)",
    )
    gpu_gint_api, wall_gint_api = time_one(
        lambda: run_gint_indirect(state, gint_reward, executor),
        warmup=args.warmup, repeat=args.repeat,
        label="gint execute_indirect (re-traces)",
    )
    gpu_gint_prep, wall_gint_prep = time_one(
        lambda: prepared.launch(),
        warmup=args.warmup, repeat=args.repeat,
        label="gint prepared.launch() (1 cuLaunch)",
    )
    print()
    print("Summary (per-iter = one full sweep over 20 tasks × 128 scenes):")
    print(f"  eager-serial          GPU kernel time: {gpu_eager:8.1f} us  "
          f"({gpu_eager / NUM_TASKS:6.2f} us per task)")
    print(f"  gint execute_indirect GPU kernel time: {gpu_gint_api:8.1f} us  "
          f"({gpu_gint_api / NUM_TASKS:6.2f} us per task)")
    print(f"  gint prepared.launch  GPU kernel time: {gpu_gint_prep:8.1f} us  "
          f"({gpu_gint_prep / NUM_TASKS:6.2f} us per task)")
    print()
    print(f"  speedup vs eager (execute_indirect API): {gpu_eager / gpu_gint_api:6.2f}x gpu, "
          f"{wall_eager / wall_gint_api:6.2f}x wall")
    print(f"  speedup vs eager (prepared launch)     : {gpu_eager / gpu_gint_prep:6.2f}x gpu, "
          f"{wall_eager / wall_gint_prep:6.2f}x wall")


if __name__ == "__main__":
    main()
