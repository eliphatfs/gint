"""
FX Graph to gint bytecode compiler.

This module handles the conversion of PyTorch FX graphs into gint bytecode programs.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Callable
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

class GintCompiler:
    """
    Compiler that converts functionalized FX graphs to gint programs using partitioning.
    """
    
    def __init__(self, gm: GraphModule, example_inputs: List[torch.Tensor]):
        self.gm = gm
        self.example_inputs = example_inputs
        
    def compile(self) -> Callable:
        from .partitioner import GraphPartitioner
        
        # Partition the graph
        partitioner = GraphPartitioner(self.gm, max_tensors=8)
        subgraph_node_lists = partitioner.partition()
        
        # Mapping from subgraph ID to compiled program
        compiled_subgraphs: Dict[int, GintCompiledSubgraph] = {}
        
        # Mapping from FX Node to its result tensor (used during runtime)
        # In a real inductor backend, we'd generate code. Here we return a callable.
        
        for i, nodes in enumerate(subgraph_node_lists):
            # For POC robustness, we make ALL intermediate results in a subgraph an "output"
            # so we can reload them and avoid complex stack management.
            all_intermediate = []
            for node in nodes:
                is_used_internally = False
                for user in node.users:
                    if user in nodes:
                        is_used_internally = True
                        break
                if is_used_internally:
                    all_intermediate.append(node)
            
            compiled_subgraphs[i] = self._compile_subgraph(nodes)
            
        def execute_wrapper(*args):
            # args are the initial placeholder values
            node_results: Dict[Node, torch.Tensor] = {}
            
            placeholder_idx = 0
            for node in self.gm.graph.nodes:
                if node.op == 'placeholder':
                    node_results[node] = args[placeholder_idx]
                    placeholder_idx += 1
            
            # Execute subgraphs or fall back to eager
            subgraph_map = {}
            for sg_id, nodes in enumerate(subgraph_node_lists):
                for node in nodes:
                    subgraph_map[node] = sg_id
            
            for node in self.gm.graph.nodes:
                if node.op == 'placeholder' or node.op == 'output':
                    continue
                
                if node in subgraph_map:
                    sg_id = subgraph_map[node]
                    # If this is the first node of the subgraph, execute the whole subgraph
                    if node == subgraph_node_lists[sg_id][0]:
                        self._execute_gint_subgraph(sg_id, subgraph_node_lists[sg_id], compiled_subgraphs[sg_id], node_results)
                else:
                    # Fallback to eager execution
                    node_results[node] = self._execute_eager(node, node_results)
            
            # Extract outputs
            for node in self.gm.graph.nodes:
                if node.op == 'output':
                    # output args are (res,)
                    res = node.args[0]
                    if isinstance(res, (list, tuple)):
                        return tuple(node_results[n] if isinstance(n, Node) else n for n in res)
                    return node_results[res] if isinstance(res, Node) else res
                    
        return execute_wrapper

    def _compile_subgraph(self, nodes: List[Node]) -> GintCompiledSubgraph:
        from ..kernel.interpreter.main import INSNS
        bytecode = []
        
        # Identify inputs and outputs of the subgraph
        inputs = []
        seen_inputs = set()
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, Node) and arg not in nodes and arg not in seen_inputs:
                    inputs.append(arg)
                    seen_inputs.add(arg)
        
        outputs = list(nodes) # For POC, every node is an output to simplify reloads
        
        # Map nodes to indices for fldg/fstg
        tensor_map = {node: i for i, node in enumerate(inputs + outputs)}

        
        from ..kernel.interpreter.main import (
            INSNS, LoadGlobalF32, LoadImm, StoreGlobalF32, 
            FAdd, FMul, FRSub, FRDiv, Halt, Dup, Pop
        )
        
        def emit(op_type, imm=0):
            bytecode.append([INSNS[op_type], imm])

        for node in nodes:
            # Inputs for this node
            for arg in node.args:
                if isinstance(arg, Node):
                    if arg in tensor_map:
                        emit(LoadGlobalF32, 16 * 0 + tensor_map[arg])
                    else:
                        # Should not happen if partitioner is correct
                        raise RuntimeError(f"Node {arg} not found in tensor map")
                elif isinstance(arg, (int, float)):
                    val = np.float32(arg).view(np.int32).item()
                    emit(LoadImm, val)
            
            # Emit instruction
            target = node.target
            if target == torch.ops.aten.add.Tensor:
                emit(FAdd)
            elif target == torch.ops.aten.mul.Tensor:
                emit(FMul)
            elif target == torch.ops.aten.sub.Tensor:
                emit(FRSub)
            elif target == torch.ops.aten.div.Tensor:
                emit(FRDiv)
            
            # Store result immediately as it's an output
            emit(StoreGlobalF32, 16 * 0 + tensor_map[node])
            # If it's NOT an output, it stays on stack for next op.
            # This is fragile for multi-use.
            # REAL FIX for POC: make EVERY node in the subgraph an "output" (temporary global storage).
            # This avoids stack management complexity.
        
        emit(Halt)
        
        # Determine shape/numel from nodes
        numel = 0
        for node in inputs + outputs:
            if hasattr(node, 'meta') and 'val' in node.meta and hasattr(node.meta['val'], 'shape'):
                numel = node.meta['val'].numel()
                break
            if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                numel = node.meta['tensor_meta'].shape.numel()
                break
        
        # Fallback if no shape info (scalar?)
        if numel == 0:
            numel = 1

        num_blocks = (numel + 31) // 32
        
        tensor_infos = []
        for _ in range(len(inputs) + len(outputs)):
            info = ProgramTensorInfo(
                elm_size=4,
                thread_stride=1,
                thread_size=numel, 
                block_strides=[32],
                block_sizes=[num_blocks],
                block_thread_offset_strides=[32]
            )
            tensor_infos.append(info)
            
        return GintCompiledSubgraph(bytecode, tensor_infos, len(inputs))

        return GintCompiledSubgraph(bytecode, tensor_infos, len(inputs))

    def _execute_gint_subgraph(self, sg_id: int, nodes: List[Node], compiled: GintCompiledSubgraph, node_results: Dict[Node, torch.Tensor]):
        # Collect inputs
        inputs = []
        seen_inputs = set()
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, Node) and arg not in nodes and arg not in seen_inputs:
                    inputs.append(arg)
                    seen_inputs.add(arg)
        
        outputs = list(nodes)
        input_tensors = [node_results[n] for n in inputs]
        
        # Prepare output tensors
        output_tensors = []
        for node in outputs:
            # Reuse input tensor properties for output creation
            # If input_tensors is empty (e.g. constant only subgraph?), use the first available tensor reference
            # safely handle the empty input case by checking inputs
            ref_tensor = input_tensors[0] if input_tensors else node_results[next(iter(node_results))]
            out = torch.empty_like(ref_tensor)
            output_tensors.append(out)
            node_results[node] = out

            
        # Execute
        all_args = input_tensors + output_tensors
        
        numel = all_args[0].numel()
        num_blocks = (numel + 31) // 32
        grid_dim = (num_blocks + REG_WIDTH - 1) // REG_WIDTH
        
        # Pass current PyTorch stream to ensure synchronization
        current_stream = torch.cuda.current_stream()
        compiled(*all_args, grid_dim=grid_dim, cuda_stream=current_stream.cuda_stream)


    def _execute_eager(self, node: Node, node_results: Dict[Node, torch.Tensor]):
        args = tuple(node_results[arg] if isinstance(arg, Node) else arg for arg in node.args)
        kwargs = {k: node_results[v] if isinstance(v, Node) else v for k, v in node.kwargs.items()}
        return node.target(*args, **kwargs)

