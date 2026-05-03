"""FX graph rewrite that replaces ``aten.bmm.default`` and
``getitem(linalg_inv_ex(a), 0)`` patterns with calls to gint's small
matmul / inverse kernels (``gint.host.matrix``).

The rewritten nodes don't appear in ``op_registry.OP_REGISTRY`` so the
partitioner skips them, and the compiler's per-node eager fallback path
ends up calling our wrappers — which dispatch to the gint kernel
directly. This keeps the partitioner / pointwise-codegen logic
unchanged.

Constraints (size N <= 4 on the trailing two dims):
  - bmm: ``aten.bmm.default(a, b)`` where the FX-meta shape ends in
    ``(N, N)`` for both args, with N matching.
  - inv: ``operator.getitem(linalg_inv_ex(a), 0)`` where ``a`` is square
    with N <= 4 in the trailing dims.
"""

import operator

import torch
from torch.fx import GraphModule, Node

from ..host.matrix import gint_bmm, gint_inv


def _shape_of(node: Node):
    if 'tensor_meta' in node.meta:
        return tuple(node.meta['tensor_meta'].shape)
    val = node.meta.get('val')
    if val is not None and hasattr(val, 'shape'):
        return tuple(val.shape)
    return None


def _is_cuda(node: Node) -> bool:
    """True iff the node's FX meta says it lives on a CUDA device.

    The gint kernels read raw CUDA pointers; rewriting CPU-side
    bmm/inv to gint would crash at runtime in ``_convert_arg`` with
    ``__cuda_array_interface__ not found``.
    """
    tm = node.meta.get('tensor_meta')
    if tm is not None and getattr(tm, 'device', None) is not None:
        return tm.device.type == 'cuda'
    val = node.meta.get('val')
    if val is not None and hasattr(val, 'device'):
        return val.device.type == 'cuda'
    return False


def _is_supported_bmm(node: Node) -> bool:
    if node.op != 'call_function' or node.target is not torch.ops.aten.bmm.default:
        return False
    if len(node.args) < 2:
        return False
    a, b = node.args[0], node.args[1]
    if not (isinstance(a, Node) and isinstance(b, Node)):
        return False
    if not (_is_cuda(a) and _is_cuda(b)):
        return False
    sa = _shape_of(a)
    sb = _shape_of(b)
    if sa is None or sb is None or len(sa) < 2 or len(sb) < 2:
        return False
    if sa[-1] != sa[-2] or sb[-1] != sb[-2] or sa[-1] != sb[-1]:
        return False
    return sa[-1] <= 4


def _is_supported_inv_getitem(node: Node) -> bool:
    if node.op != 'call_function' or node.target is not operator.getitem:
        return False
    if len(node.args) != 2 or node.args[1] != 0:
        return False
    src = node.args[0]
    if not isinstance(src, Node) or src.op != 'call_function':
        return False
    if src.target is not torch.ops.aten.linalg_inv_ex.default:
        return False
    inp = src.args[0] if src.args else None
    if not isinstance(inp, Node):
        return False
    if not _is_cuda(inp):
        return False
    s = _shape_of(inp)
    if s is None or len(s) < 2 or s[-1] != s[-2]:
        return False
    return s[-1] <= 4


def apply_special_op_rewrites(gm: GraphModule) -> GraphModule:
    """In-place rewrite of *gm* replacing supported bmm / inv patterns."""
    graph = gm.graph
    changed = False

    for node in list(graph.nodes):
        if _is_supported_bmm(node):
            with graph.inserting_before(node):
                new = graph.call_function(gint_bmm, args=(node.args[0], node.args[1]))
            new.meta = dict(node.meta)
            node.replace_all_uses_with(new)
            graph.erase_node(node)
            changed = True

    # Rewrite inv after bmm so we iterate over the (possibly modified) graph
    # cleanly. Track which linalg_inv_ex source nodes get fully replaced so
    # we can erase them once dead.
    inv_sources_to_check: list[Node] = []
    for node in list(graph.nodes):
        if _is_supported_inv_getitem(node):
            src = node.args[0]
            inp = src.args[0]
            with graph.inserting_before(node):
                new = graph.call_function(gint_inv, args=(inp,))
            new.meta = dict(node.meta)
            node.replace_all_uses_with(new)
            graph.erase_node(node)
            inv_sources_to_check.append(src)
            changed = True

    for src in inv_sources_to_check:
        if not src.users:
            graph.erase_node(src)

    if changed:
        graph.lint()
        gm.recompile()
    return gm
