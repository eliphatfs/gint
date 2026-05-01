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
from .op_registry import OpDescriptor, OpKind, get_op_descriptor, resolve_arg_order


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
# Virtual-stack code generator
# ---------------------------------------------------------------------------

class StackCodegen:
    """
    Tracks a virtual stack of FX nodes and emits bytecode via the frontend module.

    All fe.* calls are routed through the active _frontend_state context, so the
    compiler reuses exactly the same instruction-encoding path as hand-written
    sugar programs.
    """

    def __init__(self, max_stack: int, tensor_map: Dict[Node, int]):
        self.max_stack = max_stack
        self.tensor_map = tensor_map
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
            # Not on virtual stack: load from global memory.
            if arg not in self.tensor_map:
                raise RuntimeError(f"Node {arg.name} not on stack and has no global tensor entry")
            fe.fldg_1d(0, self.tensor_map[arg])
            self.vstack.append(arg)

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
            if arg in self.tensor_map:
                # Cheaper to reload from global than shuffle the stack (avoids a
                # Swap that would displace the current top for later ops).
                fe.fldg_1d(0, self.tensor_map[arg])
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
            # Depth >= 2: reload from global if possible, otherwise error.
            if arg in self.tensor_map:
                fe.fldg_1d(0, self.tensor_map[arg])
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

        # Handle result: store to global and/or keep/discard from virtual stack.
        internal_uses = self.uses_left.get(node, 0)
        in_global     = node in self.tensor_map

        if in_global:
            if internal_uses > 0:
                # Write to global AND keep on stack for future internal consumers.
                # Dup first so the store doesn't consume our only copy.
                fe.dup()
                self.vstack.append(node)     # virtual: result×2
                fe.fstg_1d(0, self.tensor_map[node])
                self.vstack.pop()            # store pops one → result×1 remains
            else:
                # Last internal use (or no internal use): just store and remove.
                fe.fstg_1d(0, self.tensor_map[node])
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

    def _required_globals(self, nodes: List[Node]) -> Set[Node]:
        """Nodes that need a global tensor slot (external inputs/outputs + multi-use intermediates)."""
        node_set = set(nodes)
        required: Set[Node] = set()
        for i, node in enumerate(nodes):
            for arg in node.args:
                if isinstance(arg, Node) and arg not in node_set:
                    required.add(arg)    # external input
            is_ext_out = node in self.graph_outputs or any(u not in node_set for u in node.users)
            later_uses = [u for u in node.users if u in nodes[i + 1:]]
            if is_ext_out or len(later_uses) > 1:
                required.add(node)
        return required

    def _stack_fits(self, nodes: List[Node]) -> bool:
        """Simulate a conservative stack depth to check the subgraph fits.

        We track the virtual depth (items logically on the stack) and also
        account for the peak internal depth that custom multi-instruction ops
        need during their emission (peak_stack_extra).
        """
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
        global_nodes = self._sorted_globals(partitioner._required_globals(nodes))
        tensor_map = {n: i for i, n in enumerate(global_nodes)}

        # Open a FrontendState context so all fe.* calls accumulate into fe_state.bc.
        fe_state = FrontendState([])
        token = _frontend_state.set(fe_state)
        try:
            codegen = StackCodegen(max_stack=8, tensor_map=tensor_map)
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

        # Compute broadcast plan for per-tensor info.
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
        num_inner_blocks = plan['num_inner_blocks']

        tensor_infos = []
        for i, node in enumerate(global_nodes):
            entry = plan['per_tensor'][i]
            tensor_infos.append(ProgramTensorInfo(
                elm_size=4,
                batch_strides=entry['batch_strides'],
                batch_shape=list(plan['batch_dims']),
                block_shape_stride_1=[plan['block_size'], entry['block_stride']],
                block_shape_stride_2=[1, 0],
                block_grid_dims=[num_inner_blocks, 1],
                block_grid_steps=[128, 1],
            ))

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
                                     grid_dim=plan['grid_dim'],
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
        input_node = node.args[0]
        input_shape = list(partitioner._get_shape(input_node))
        output_shape = list(partitioner._get_shape(node))

        reduction_size = input_shape[-1]
        n_chunks = math.ceil(reduction_size / 128)

        # Determine tensor slot ordering — input first, output second
        global_nodes = self._sorted_globals(partitioner._required_globals([node]))
        tensor_map = {n: i for i, n in enumerate(global_nodes)}
        input_slot = tensor_map[input_node]
        output_slot = tensor_map[node]

        op_desc = get_op_descriptor(node)
        combine_fn = op_desc.combine_fn

        fe_state = FrontendState([])
        token = _frontend_state.set(fe_state)
        try:
            # Load and accumulate chunks using the reduction's combine op.
            fe.fldg_1d(0, input_slot)
            for k in range(1, n_chunks):
                fe.fldg_1d(k * 128, input_slot)
                combine_fn()
            # Warp reduce + width-lane combine
            op_desc.emit_fn(node)
            # Store
            fe.fstg_1d(0, output_slot)
            fe.halt()
            bytecode = fe_state.bc
        finally:
            _frontend_state.reset(token)

        # Build tensor infos for each global node
        batch_dims = input_shape[:-1] if len(input_shape) > 1 else []
        tensor_infos = []
        for gn in global_nodes:
            if gn is input_node:
                bstrides = _c_contiguous_strides(input_shape)[:-1] if len(input_shape) > 1 else []
                tensor_infos.append(ProgramTensorInfo(
                    elm_size=4,
                    batch_strides=bstrides,
                    batch_shape=list(batch_dims),
                    block_shape_stride_1=[reduction_size, 1],
                    block_shape_stride_2=[1, 0],
                    block_grid_dims=[1, 1],
                    block_grid_steps=[128, 1],
                ))
            else:  # output node
                keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get('keepdim', False)
                if keepdim:
                    out_batch = output_shape[:-1] if len(output_shape) > 1 else []
                else:
                    out_batch = list(output_shape) if output_shape else []
                bstrides = _c_contiguous_strides(out_batch) if out_batch else []
                tensor_infos.append(ProgramTensorInfo(
                    elm_size=4,
                    batch_strides=bstrides,
                    batch_shape=list(batch_dims),
                    block_shape_stride_1=[1, 1],
                    block_shape_stride_2=[1, 0],
                    block_grid_dims=[1, 1],
                    block_grid_steps=[1, 1],
                ))

        grid_dim = max(1, int(np.prod(batch_dims))) if batch_dims else 1

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
        """
        scalar_prefix, broadcast_suffix = GintCompiler._split_pointwise_chain(
            reduction_node, pw_schedule)

        reduction_input = reduction_node.args[0]
        input_shape = list(partitioner._get_shape(reduction_input))
        reduction_size = input_shape[-1]
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
            t = torch.empty(compiled.output_shape, dtype=ref.dtype, device=ref.device)
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
