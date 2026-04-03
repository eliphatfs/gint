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
from .op_registry import OpDescriptor, OpKind, get_op_descriptor


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


def _compute_broadcast_plan(output_shape, tensor_shapes):
    """Compute a broadcast plan mapping output_shape to gint block+batch decomposition.

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
    for p in padded:
        c_strides = _c_contiguous_strides(p)

        # block_stride: 0 if any merged inner dim broadcasts, 1 otherwise
        inner_broadcast = any(
            p[d] == 1 and output_shape[d] > 1
            for d in range(ndim - merge_count, ndim)
        )
        block_stride = 0 if inner_broadcast else 1

        # batch_strides: 0 for broadcast dims, C-contiguous stride otherwise
        b_strides = []
        for d in range(len(batch_dims)):
            if p[d] == 1 and output_shape[d] > 1:
                b_strides.append(0)
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
                 output_shape: tuple = (), grid_dim: int = 1):
        super().__init__()
        self.bytecode = bytecode
        self.tensor_infos = tensor_infos
        self.output_shape = output_shape
        self.grid_dim = grid_dim

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
        self.uses_left: Dict[Node, int] = {}
        # Nodes that must be stored to global memory (external outputs or
        # multi-use intermediates already registered in tensor_map).
        self.ext_out: Set[Node] = set()

    def set_subgraph_nodes(self, nodes: List[Node], graph_outputs: Set[Node]):
        node_set = set(nodes)
        self.uses_left = {}
        self.ext_out = set()
        for node in nodes:
            internal = sum(1 for u in node.users if u in node_set)
            self.uses_left[node] = internal
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

        if depth == -1:
            # Not on virtual stack: load from global memory.
            if arg not in self.tensor_map:
                raise RuntimeError(f"Node {arg.name} not on stack and has no global tensor entry")
            fe.fldg_1d(0, self.tensor_map[arg])
            self.vstack.append(arg)

        elif depth == 0:
            # Already on top.  Duplicate if it will be needed again after this use.
            if uses_left > 1:
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

    def emit_op(self, node: Node, op_desc: OpDescriptor):
        """
        Emit the operation, update the virtual stack (pop inputs, push result),
        then handle result storage / cleanup.
        """
        # Emit the instruction(s) for this operation.
        op_desc.emit_fn(node)

        # Pop consumed inputs from the virtual stack.
        for _ in range(op_desc.arity):
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

    def _is_supported(self, node: Node) -> bool:
        if node.op != 'call_function':
            return False
        return get_op_descriptor(node) is not None

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

            if current_shape is None:
                can_add = True
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
                    global_shapes = [self._get_shape(n) for n in self._required_globals(scheduled)]
                    global_shapes = [s for s in global_shapes if s is not None]
                    if global_shapes and any(s != merged_shape for s in global_shapes):
                        if _compute_broadcast_plan(merged_shape, global_shapes) is None:
                            can_add = False

            if can_add:
                current_shape = _broadcast_shapes(current_shape, shape) if current_shape else shape
                current.append(node)
            else:
                if current:
                    subgraphs.append(ForestTraverser(current).get_schedule())
                # Check if the rejected node can start a new subgraph on its own
                # (its globals must be broadcast-compatible with its shape).
                solo = ForestTraverser([node]).get_schedule()
                solo_globals = [self._get_shape(n) for n in self._required_globals(solo)]
                solo_globals = [s for s in solo_globals if s is not None]
                if solo_globals and any(s != shape for s in solo_globals):
                    if _compute_broadcast_plan(shape, solo_globals) is None:
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

        # Fuse reduction+broadcast+pointwise pairs where possible.
        merged_schedules: List[List[Node]] = []
        compiled: Dict[int, GintCompiledSubgraph] = {}
        i = 0
        sg_index = 0
        while i < len(raw_schedules):
            schedule = raw_schedules[i]
            if self._is_reduction_subgraph(schedule):
                reduction_node = schedule[0]
                keepdim = (reduction_node.args[2]
                           if len(reduction_node.args) > 2
                           else reduction_node.kwargs.get('keepdim', False))
                # Try to fuse with the following pointwise subgraph.
                if (keepdim and i + 1 < len(raw_schedules)
                        and not self._is_reduction_subgraph(raw_schedules[i + 1])
                        and self._can_fuse_with_reduction(
                            reduction_node, raw_schedules[i + 1])):
                    pw_schedule = raw_schedules[i + 1]
                    fused_nodes = [reduction_node] + pw_schedule
                    compiled[sg_index] = self._compile_fused_reduction_subgraph(
                        reduction_node, pw_schedule, partitioner)
                    merged_schedules.append(fused_nodes)
                    sg_index += 1
                    i += 2
                    continue
                compiled[sg_index] = self._compile_reduction_subgraph(
                    reduction_node, partitioner)
            else:
                compiled[sg_index] = self._compile_subgraph(schedule, partitioner)
            merged_schedules.append(schedule)
            sg_index += 1
            i += 1

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
    def _can_fuse_with_reduction(reduction_node: Node, pw_schedule: List[Node]) -> bool:
        """Check if a pointwise schedule can fuse with a preceding keepdim reduction.

        Currently only fuses a single binary elementwise op that consumes both the
        reduction result and the reduction input, and the reduction result has no
        external users outside the pointwise schedule.
        """
        if len(pw_schedule) != 1:
            return False
        pw_node = pw_schedule[0]
        desc = get_op_descriptor(pw_node)
        if desc is None or desc.kind != OpKind.ELEMENTWISE or desc.arity != 2:
            return False
        reduction_input = reduction_node.args[0]
        node_args = [a for a in pw_node.args if isinstance(a, Node)]
        if reduction_node not in node_args or reduction_input not in node_args:
            return False
        # Reduction result must have no users outside the fused subgraph.
        pw_set = set(pw_schedule)
        for user in reduction_node.users:
            if user not in pw_set:
                return False
        return True

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
                arg_indices = op_desc.arg_order if op_desc.arg_order is not None \
                              else list(range(len(node.args)))
                for idx in arg_indices:
                    arg = node.args[idx]
                    if isinstance(arg, (Node, int, float)):
                        codegen.handle_operand(arg)

                # Emit the operation and manage the result.
                codegen.emit_op(node, op_desc)

            fe.halt()
            bytecode = fe_state.bc
        finally:
            _frontend_state.reset(token)

        # Compute broadcast plan for per-tensor info.
        all_shapes = [partitioner._get_shape(n) for n in global_nodes]
        output_shape = all_shapes[0]
        for s in all_shapes[1:]:
            if s is not None:
                output_shape = _broadcast_shapes(output_shape, s)

        plan = _compute_broadcast_plan(output_shape, [s for s in all_shapes if s is not None])
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

        from ..host.debug import dump_bytecode
        dump_bytecode(bytecode, "tmp/sg.gint")

        return GintCompiledSubgraph(bytecode, tensor_infos,
                                     output_shape=output_shape,
                                     grid_dim=plan['grid_dim'])

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

        fe_state = FrontendState([])
        token = _frontend_state.set(fe_state)
        try:
            # Load and accumulate chunks
            fe.fldg_1d(0, input_slot)
            for k in range(1, n_chunks):
                fe.fldg_1d(k * 128, input_slot)
                fe.fadd()
            # Warp reduce + width-lane combine
            op_desc = get_op_descriptor(node)
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
    ) -> GintCompiledSubgraph:
        """Compile a fused reduction+broadcast+pointwise subgraph.

        Emits bytecode in two phases:
        Phase 1: Load input in chunks, accumulate, reduce → scalar on stack.
        Phase 2: For each chunk, reload input, apply pointwise op with scalar, store.
        """
        pw_node = pw_schedule[0]
        input_node = reduction_node.args[0]
        input_shape = list(partitioner._get_shape(input_node))
        output_shape = list(partitioner._get_shape(pw_node))

        reduction_size = input_shape[-1]
        n_chunks = math.ceil(reduction_size / 128)

        all_nodes = [reduction_node] + pw_schedule
        global_nodes = self._sorted_globals(partitioner._required_globals(all_nodes))
        tensor_map = {n: i for i, n in enumerate(global_nodes)}
        input_slot = tensor_map[input_node]
        output_slot = tensor_map[pw_node]

        # Determine whether the reduction result is args[1] of the pointwise op.
        # If so, after dup+load we need swap to match the registry's expected stack order.
        pw_args = pw_node.args
        reduction_is_second = (isinstance(pw_args[1], Node) and pw_args[1] is reduction_node)

        fe_state = FrontendState([])
        token = _frontend_state.set(fe_state)
        try:
            # Phase 1: Load and reduce
            fe.fldg_1d(0, input_slot)
            for k in range(1, n_chunks):
                fe.fldg_1d(k * 128, input_slot)
                fe.fadd()
            red_desc = get_op_descriptor(reduction_node)
            red_desc.emit_fn(reduction_node)
            # Scalar reduction result is now on top of stack.

            # Phase 2: For each chunk, apply pointwise op
            pw_desc = get_op_descriptor(pw_node)
            for k in range(n_chunks):
                fe.dup()                              # keep scalar for next iteration
                fe.fldg_1d(k * 128, input_slot)       # load input chunk
                if reduction_is_second:
                    # Stack: [scalar_copy, x_chunk]. Registry expects top=args[1]=scalar.
                    fe.swap()
                # else: reduction is args[0], registry expects top=args[1]=x. Already correct.
                pw_desc.emit_fn(pw_node)
                fe.fstg_1d(k * 128, output_slot)
            fe.pop()                                  # discard remaining scalar
            fe.halt()
            bytecode = fe_state.bc
        finally:
            _frontend_state.reset(token)

        # Build tensor infos — both input and output use the full reduction dim.
        batch_dims = input_shape[:-1] if len(input_shape) > 1 else []
        bstrides = _c_contiguous_strides(input_shape)[:-1] if len(input_shape) > 1 else []

        tensor_infos = []
        for gn in global_nodes:
            tensor_infos.append(ProgramTensorInfo(
                elm_size=4,
                batch_strides=list(bstrides),
                batch_shape=list(batch_dims),
                block_shape_stride_1=[reduction_size, 1],
                block_shape_stride_2=[1, 0],
                block_grid_dims=[1, 1],
                block_grid_steps=[128, 1],
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
