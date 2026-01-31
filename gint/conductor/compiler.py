"""
FX Graph to gint bytecode compiler.

This module handles the conversion of PyTorch FX graphs into gint bytecode programs.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Callable, Set, Optional
from torch.fx import GraphModule, Node

from ..host.executor import (
    BaseExecutableProgram, 
    ProgramData, 
    ProgramTensorInfo, 
    TensorInterface,
    _convert_arg
)
from ..host import frontend as fe
from ..kernel.interpreter.main import REG_WIDTH

class GintCompiledSubgraph(BaseExecutableProgram):
    """
    A compiled gint subgraph.
    """
    def __init__(self, bytecode: List[List[int]], tensor_infos: List[ProgramTensorInfo], num_inputs: int):
        super().__init__()
        self.bytecode = bytecode
        self.tensor_infos = tensor_infos
        self.num_inputs = num_inputs
    
    def get_program(self, *args: TensorInterface, **extra_kwargs) -> ProgramData:
        bc_array = np.array(self.bytecode, dtype=np.int32).reshape(-1)
        return ProgramData(bc_array, self.tensor_infos)

class GraphPartitioner:
    """
    Partitions an FX graph into subgraphs compatible with gint.
    
    Constraints:
    - Supported ops: add, sub, mul, div.
    - Max 8 global tensors per subgraph (External Inputs + External Outputs + Multi-use Intermediates).
    - All tensors in a subgraph must have the same shape.
    - Stack depth must not exceed 8.
    """
    
    SUPPORTED_OPS = {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Tensor,
    }

    def __init__(self, gm: GraphModule, max_tensors: int = 8, max_stack: int = 8):
        self.gm = gm
        self.max_tensors = max_tensors
        self.max_stack = max_stack
        
        # Pre-pass: final outputs
        self.graph_outputs = set()
        for node in self.gm.graph.nodes:
            if node.op == 'output':
                for arg in node.args:
                    if isinstance(arg, Node):
                        self.graph_outputs.add(arg)
                    elif isinstance(arg, (list, tuple)):
                        for a in arg:
                            if isinstance(a, Node):
                                self.graph_outputs.add(a)

    def _get_shape(self, node: Node) -> Optional[torch.Size]:
        if 'tensor_meta' in node.meta:
            return node.meta['tensor_meta'].shape
        if 'val' in node.meta and hasattr(node.meta['val'], 'shape'):
            return node.meta['val'].shape
        return None

    def partition(self) -> List[List[Node]]:
        subgraphs = []
        current_nodes = []
        current_shape = None
        
        for node in self.gm.graph.nodes:
            if node.op in ('placeholder', 'output'):
                continue
                
            supported = node.op == 'call_function' and node.target in self.SUPPORTED_OPS
            shape = self._get_shape(node) if supported else None
            
            if not supported or shape is None:
                if current_nodes:
                    subgraphs.append(current_nodes)
                    current_nodes = []
                    current_shape = None
                continue
                
            can_add = True
            if current_shape is not None and shape != current_shape:
                can_add = False
            
            if can_add:
                potential = current_nodes + [node]
                # Check Resource Limits (Tensors and Stack)
                ext_tensors = self._get_required_global_tensors(potential)
                if len(ext_tensors) > self.max_tensors:
                    can_add = False
                elif not self._simulate_stack(potential):
                    can_add = False
            
            if can_add:
                if current_shape is None:
                    current_shape = shape
                current_nodes.append(node)
            else:
                if current_nodes:
                    subgraphs.append(current_nodes)
                current_nodes = [node]
                current_shape = shape
        
        if current_nodes:
            subgraphs.append(current_nodes)
            
        return subgraphs

    def _get_required_global_tensors(self, nodes: List[Node]) -> Set[Node]:
        node_set = set(nodes)
        required = set()
        for i, node in enumerate(nodes):
            # Inputs from outside
            for arg in node.args:
                if isinstance(arg, Node) and arg not in node_set:
                    required.add(arg)
            
            # Outputs (external or multi-use)
            is_ext_out = False
            if node in self.graph_outputs:
                is_ext_out = True
            else:
                for user in node.users:
                    if user not in node_set:
                        is_ext_out = True
                        break
            
            uses_in_sub = [u for u in node.users if u in nodes[i+1:]]
            if is_ext_out or len(uses_in_sub) > 1:
                required.add(node)
        return required

    def _simulate_stack(self, nodes: List[Node]) -> bool:
        stack = [] # List of Nodes
        node_set = set(nodes)
        
        # Policy: 
        # - Load if not top.
        # - Store if multi-use or external.
        # - Keep on stack if single-use by NEXT.
        
        for i, node in enumerate(nodes):
            args = [a for a in node.args if isinstance(a, Node)]
            # Load phase
            for arg in args:
                if not (stack and stack[-1] == arg):
                    stack.append(arg)
                else:
                    # Arg is at top, will be consumed
                    stack.pop()
            
            if len(stack) > self.max_stack: return False
            
            # Op consumes any newly loaded args too?
            # Actually, opacity in simulation: peak is before OP if many loads.
            # Pop and push result
            stack.append(node)
            if len(stack) > self.max_stack: return False
            
            # Determine if node stays on stack
            is_ext_out = node in self.graph_outputs or any(u not in node_set for u in node.users)
            uses_in_sub = [u for u in node.users if u in nodes[i+1:]]
            
            # If not kept on stack, pop it (simulating Store)
            keep_on_stack = len(uses_in_sub) == 1 and i+1 < len(nodes) and nodes[i+1] in uses_in_sub and not is_ext_out
            if not keep_on_stack:
                stack.pop()
                
        return True

class GintCompiler:
    """
    Compiler that converts functionalized FX graphs to gint programs using partitioning and stack optimization.
    """
    
    def __init__(self, gm: GraphModule, example_inputs: List[torch.Tensor]):
        self.gm = gm
        self.example_inputs = example_inputs
        
    def compile(self) -> Callable:
        # Partition the graph
        partitioner = GraphPartitioner(self.gm, max_tensors=8, max_stack=8)
        subgraph_node_lists = partitioner.partition()
        
        compiled_subgraphs: Dict[int, GintCompiledSubgraph] = {}
        
        for i, nodes in enumerate(subgraph_node_lists):
            compiled_subgraphs[i] = self._compile_subgraph(nodes)
            
        def execute_wrapper(*args):
            node_results: Dict[Node, torch.Tensor] = {}
            
            placeholder_idx = 0
            for node in self.gm.graph.nodes:
                if node.op == 'placeholder':
                    node_results[node] = args[placeholder_idx]
                    placeholder_idx += 1
            
            subgraph_map = {}
            for sg_id, nodes in enumerate(subgraph_node_lists):
                for node in nodes:
                    subgraph_map[node] = sg_id
            
            for node in self.gm.graph.nodes:
                if node.op in ('placeholder', 'output'):
                    continue
                
                if node in subgraph_map:
                    sg_id = subgraph_map[node]
                    if node == subgraph_node_lists[sg_id][0]:
                        self._execute_gint_subgraph(sg_id, subgraph_node_lists[sg_id], compiled_subgraphs[sg_id], node_results)
                else:
                    node_results[node] = self._execute_eager(node, node_results)
            
            for node in self.gm.graph.nodes:
                if node.op == 'output':
                    res = node.args[0]
                    if isinstance(res, (list, tuple)):
                        return tuple(node_results[n] if isinstance(n, Node) else n for n in res)
                    return node_results[res] if isinstance(res, Node) else res
                    
        return execute_wrapper

    def _compile_subgraph(self, nodes: List[Node]) -> GintCompiledSubgraph:
        from ..kernel.interpreter.main import (
            INSNS, LoadGlobalF32, LoadImm, StoreGlobalF32, 
            FAdd, FMul, FRSub, FRDiv, Halt, Dup, Pop
        )
        bytecode = []
        
        node_set = set(nodes)
        partitioner = GraphPartitioner(self.gm) # Reuse logic
        global_nodes = sorted(list(partitioner._get_required_global_tensors(nodes)), key=lambda n: n.name if hasattr(n, 'name') else str(id(n)))
        tensor_map = {node: i for i, node in enumerate(global_nodes)}
        
        # Identify external inputs specifically for execute_wrapper
        inputs = [n for n in global_nodes if n not in node_set]
        
        def emit(op_type, imm=0):
            bytecode.append([INSNS[op_type], imm])

        stack = [] # Simulating gint stack content
        
        for i, node in enumerate(nodes):
            # 1. Prepare Arguments
            for arg in node.args:
                if isinstance(arg, Node):
                    if stack and stack[-1] == arg:
                        # Top of stack, use it
                        stack.pop()
                    else:
                        # Load from global storage
                        if arg in tensor_map:
                            emit(LoadGlobalF32, 16 * 0 + tensor_map[arg])
                        else:
                            raise RuntimeError(f"Required node {arg} not found in global storage")
                elif isinstance(arg, (int, float)):
                    val = np.float32(arg).view(np.int32).item()
                    emit(LoadImm, val)
            
            # 2. Emit Operation
            target = node.target
            if target == torch.ops.aten.add.Tensor:
                emit(FAdd)
            elif target == torch.ops.aten.mul.Tensor:
                emit(FMul)
            elif target == torch.ops.aten.sub.Tensor:
                emit(FRSub)
            elif target == torch.ops.aten.div.Tensor:
                emit(FRDiv)
            
            # 3. Post-op: Handle result
            is_ext_out = node in partitioner.graph_outputs or any(u not in node_set for u in node.users)
            uses_in_sub = [u for u in node.users if u in nodes[i+1:]]
            
            # If used by next node only, and NOT an external output, keep on stack.
            # Otherwise, Store it.
            keep_on_stack = len(uses_in_sub) == 1 and i+1 < len(nodes) and nodes[i+1] in uses_in_sub and not is_ext_out
            
            if is_ext_out or len(uses_in_sub) > 1:
                if keep_on_stack:
                    emit(Dup)
                    emit(StoreGlobalF32, 16 * 0 + tensor_map[node])
                    stack.append(node)
                else:
                    emit(StoreGlobalF32, 16 * 0 + tensor_map[node])
                    # No longer on stack (Store pops)
            elif keep_on_stack:
                stack.append(node)
            else:
                # Not used or used multiple times but not next (should be handled by Store above)
                # If not stored and not kept, we pop it to keep stack clean if it's dead.
                # In functional graphs, this shouldn't really happen for intermediates unless they are unused.
                if not uses_in_sub and not is_ext_out:
                    emit(Pop)

        emit(Halt)
        
        # Calculate numel
        numel = 0
        some_node = global_nodes[0] if global_nodes else nodes[0]
        if hasattr(some_node, 'meta') and 'val' in some_node.meta and hasattr(some_node.meta['val'], 'shape'):
            numel = some_node.meta['val'].numel()
        elif hasattr(some_node, 'meta') and 'tensor_meta' in some_node.meta:
            numel = some_node.meta['tensor_meta'].shape.numel()
        
        if numel == 0: numel = 1
        num_blocks = (numel + 31) // 32
        
        tensor_infos = [
            ProgramTensorInfo(4, 1, numel, [32], [num_blocks], [32])
            for _ in range(len(global_nodes))
        ]
            
        return GintCompiledSubgraph(bytecode, tensor_infos, len(inputs))

    def _execute_gint_subgraph(self, sg_id: int, nodes: List[Node], compiled: GintCompiledSubgraph, node_results: Dict[Node, torch.Tensor]):
        node_set = set(nodes)
        partitioner = GraphPartitioner(self.gm)
        global_nodes = sorted(list(partitioner._get_required_global_tensors(nodes)), key=lambda n: n.name if hasattr(n, 'name') else str(id(n)))
        
        inputs = [n for n in global_nodes if n not in node_set]
        outputs = [n for n in global_nodes if n in node_set]
        
        input_tensors = [node_results[n] for n in inputs]
        output_tensors = []
        for node in outputs:
            # Check if this node already has a result (might happen if re-executing?)
            # Actually, inductor usually functionalizes so this is fine.
            ref = input_tensors[0] if input_tensors else node_results[next(iter(node_results))]
            out = torch.empty_like(ref)
            output_tensors.append(out)
            node_results[node] = out
            
        # The executor expects args in the same order as global_nodes
        # global_nodes was sorted, so we must align all_args to it.
        all_args_map = {n: t for n, t in zip(inputs + outputs, input_tensors + output_tensors)}
        all_args = [all_args_map[n] for n in global_nodes]
        
        numel = all_args[0].numel() if all_args else 1
        grid_dim = ((numel + 31) // 32 + REG_WIDTH - 1) // REG_WIDTH
        
        compiled(*all_args, grid_dim=grid_dim, cuda_stream=torch.cuda.current_stream().cuda_stream)

    def _execute_eager(self, node: Node, node_results: Dict[Node, torch.Tensor]):
        args = tuple(node_results[arg] if isinstance(arg, Node) else arg for arg in node.args)
        kwargs = {k: node_results[v] if isinstance(v, Node) else v for k, v in node.kwargs.items()}
        return node.target(*args, **kwargs)

