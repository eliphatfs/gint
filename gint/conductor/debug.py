"""Inspection utilities for the gint conductor.

Use ``inspect_subgraphs`` to compile a function through ``torch.compile`` with
the gint backend and capture the produced subgraphs together with the bytecode
emitted for each. This is the right tool when a torch.compile path looks
slower than expected or you want to verify partitioning / fusion behavior.

Example::

    from gint.conductor.debug import inspect_subgraphs, print_subgraphs

    def rms_norm(x, w):
        rstd = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-5)
        return x * rstd * w

    sgs = inspect_subgraphs(rms_norm, x, w)
    print_subgraphs(sgs)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List

import torch

from ..host.debug import pprint_bytecode
from .compiler import GintCompiler


@dataclass
class SubgraphInfo:
    """One subgraph the conductor produced for a torch.compile call."""
    kind: str                       # 'pointwise', 'reduction', 'fused-reduction'
    nodes: List[Any]                # FX nodes in the subgraph
    bytecode: List[List[int]]       # [opcode, operand] pairs
    output_shape: Any = None
    grid_dim: int = 0

    def node_summary(self) -> str:
        return "\n".join(f"  {n.op:<14} {n.name:<24} target={n.target}"
                         for n in self.nodes)

    def bytecode_pp(self) -> str:
        return pprint_bytecode(self.bytecode)


def inspect_subgraphs(fn: Callable, *args, **kwargs) -> List[SubgraphInfo]:
    """Run ``fn`` through ``torch.compile(backend='gint', options={"cuda_graphs": False})`` and
    return one SubgraphInfo per subgraph the conductor compiled.

    Disables cuda graphs so the side-stream warmup pass inside
    ``make_graphed_callables`` doesn't double-compile.
    """
    captured: List[SubgraphInfo] = []

    orig_pointwise = GintCompiler._compile_subgraph
    orig_reduction = GintCompiler._compile_reduction_subgraph
    orig_fused = GintCompiler._compile_fused_reduction_subgraph

    def cap_pointwise(self, nodes, partitioner):
        result = orig_pointwise(self, nodes, partitioner)
        captured.append(SubgraphInfo(
            kind='pointwise',
            nodes=list(nodes),
            bytecode=result.bytecode,
            output_shape=getattr(result, 'output_shape', None),
            grid_dim=getattr(result, 'grid_dim', 0),
        ))
        return result

    def cap_reduction(self, node, partitioner):
        result = orig_reduction(self, node, partitioner)
        captured.append(SubgraphInfo(
            kind='reduction',
            nodes=[node],
            bytecode=result.bytecode,
            output_shape=getattr(result, 'output_shape', None),
            grid_dim=getattr(result, 'grid_dim', 0),
        ))
        return result

    def cap_fused(self, red_node, pw_schedule, partitioner, pre_prefix=None):
        result = orig_fused(self, red_node, pw_schedule, partitioner,
                            pre_prefix=pre_prefix)
        nodes = (pre_prefix or []) + [red_node] + list(pw_schedule)
        captured.append(SubgraphInfo(
            kind='fused-reduction',
            nodes=nodes,
            bytecode=result.bytecode,
            output_shape=getattr(result, 'output_shape', None),
            grid_dim=getattr(result, 'grid_dim', 0),
        ))
        return result

    GintCompiler._compile_subgraph = cap_pointwise
    GintCompiler._compile_reduction_subgraph = cap_reduction
    GintCompiler._compile_fused_reduction_subgraph = cap_fused
    try:
        compiled = torch.compile(fn, backend='gint', options={"cuda_graphs": False})
        compiled(*args, **kwargs)
        torch.cuda.synchronize()
    finally:
        GintCompiler._compile_subgraph = orig_pointwise
        GintCompiler._compile_reduction_subgraph = orig_reduction
        GintCompiler._compile_fused_reduction_subgraph = orig_fused

    return captured


def print_subgraphs(sgs: List[SubgraphInfo], show_bytecode: bool = True) -> None:
    """Pretty-print the result of ``inspect_subgraphs`` to stdout."""
    print(f"\n=== {len(sgs)} subgraph(s) compiled ===\n")
    for i, sg in enumerate(sgs):
        print(f"--- subgraph {i}: kind={sg.kind}, nodes={len(sg.nodes)}, "
              f"shape={sg.output_shape}, grid_dim={sg.grid_dim} ---")
        print(sg.node_summary())
        if show_bytecode:
            print(f"  bytecode ({len(sg.bytecode)} insns):")
            for line in sg.bytecode_pp().split("\n"):
                print(f"    {line}")
        print()
