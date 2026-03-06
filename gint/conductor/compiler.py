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
from .op_registry import OpDescriptor, get_op_descriptor


# ---------------------------------------------------------------------------
# Compiled-subgraph wrapper
# ---------------------------------------------------------------------------

class GintCompiledSubgraph(BaseExecutableProgram):
    """An already-compiled gint subgraph ready for execution."""

    def __init__(self, bytecode: List[List[int]], tensor_infos: List[ProgramTensorInfo]):
        super().__init__()
        self.bytecode = bytecode
        self.tensor_infos = tensor_infos

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

            shape = self._get_shape(node)
            if shape is None:
                if current:
                    subgraphs.append(ForestTraverser(current).get_schedule())
                    current = []
                    current_shape = None
                continue

            can_add = (current_shape is None or shape == current_shape)
            if can_add:
                candidate = current + [node]
                scheduled = ForestTraverser(candidate).get_schedule()
                if len(self._required_globals(scheduled)) > self.max_tensors:
                    can_add = False
                elif not self._stack_fits(scheduled):
                    can_add = False

            if can_add:
                if current_shape is None:
                    current_shape = shape
                current.append(node)
            else:
                if current:
                    subgraphs.append(ForestTraverser(current).get_schedule())
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
        subgraph_schedules = partitioner.partition()

        compiled: Dict[int, GintCompiledSubgraph] = {}
        for i, schedule in enumerate(subgraph_schedules):
            compiled[i] = self._compile_subgraph(schedule, partitioner)

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

        # Build tensor infos (all uniform 1-D for element-wise ops).
        numel = self._numel(global_nodes, nodes)
        num_blocks = max(1, (numel + 127) // 128)
        tensor_infos = [
            ProgramTensorInfo(
                elm_size=4,
                batch_strides=[],
                batch_shape=[],
                block_shape_stride_1=[numel, 1],
                block_shape_stride_2=[1, 0],
                block_grid_dims=[num_blocks, 1],
                block_grid_steps=[128, 1],
            )
            for _ in global_nodes
        ]

        from ..host.debug import dump_bytecode
        dump_bytecode(bytecode, "tmp/sg.gint")

        return GintCompiledSubgraph(bytecode, tensor_infos)

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
            t = torch.empty_like(ref)
            output_tensors.append(t)
            results[node] = t

        all_tensors = {n: t for n, t in zip(inputs + outputs, input_tensors + output_tensors)}
        ordered     = [all_tensors[n] for n in global_nodes]

        numel    = ordered[0].numel() if ordered else 1
        grid_dim = max(1, (numel + 127) // 128)

        compiled(*ordered, grid_dim=grid_dim,
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
