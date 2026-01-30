import torch
from torch.fx import GraphModule, Node
from typing import Dict, List, Set, Optional, Tuple


class GraphPartitioner:
    """
    Partitions an FX graph into subgraphs compatible with gint.
    
    Constraints:
    - Supported ops: add, sub, mul, div.
    - Max 8 tensors per subgraph (including inputs and outputs).
    - All tensors in a subgraph must have the same shape.
    """
    
    SUPPORTED_OPS = {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Tensor,
    }

    def __init__(self, gm: GraphModule, max_tensors: int = 8):
        self.gm = gm
        self.max_tensors = max_tensors
        self.node_to_subgraph: Dict[Node, int] = {}
        self.subgraphs: List[List[Node]] = []

    def _get_shape(self, node: Node) -> Optional[torch.Size]:
        if 'tensor_meta' in node.meta:
            return node.meta['tensor_meta'].shape
        if 'val' in node.meta and hasattr(node.meta['val'], 'shape'):
            return node.meta['val'].shape
        return None

    def _is_supported(self, node: Node) -> bool:
        if node.op != 'call_function':
            return False
        return node.target in self.SUPPORTED_OPS

    def partition(self) -> List[List[Node]]:
        """
        Greedily partition nodes into compatible subgraphs.
        """
        current_subgraph_nodes = []
        current_shape = None
        
        for node in self.gm.graph.nodes:
            if node.op in ('placeholder', 'output'):
                continue
                
            if not self._is_supported(node):
                if current_subgraph_nodes:
                    self.subgraphs.append(current_subgraph_nodes)
                    current_subgraph_nodes = []
                    current_shape = None
                continue
                
            shape = self._get_shape(node)
            if shape is None:
                # Fallback if no shape info
                if current_subgraph_nodes:
                    self.subgraphs.append(current_subgraph_nodes)
                    current_subgraph_nodes = []
                    current_shape = None
                continue

            # Check if compatible with current subgraph
            can_add = True
            if current_shape is not None and shape != current_shape:
                can_add = False
            
            if can_add:
                # Unique tensors = inputs from outside subgraph + all outputs in subgraph
                potential_nodes = current_subgraph_nodes + [node]
                inputs = set()
                outputs = set()
                for n in potential_nodes:
                    outputs.add(n)
                    for arg in n.args:
                        if isinstance(arg, Node) and arg not in potential_nodes:
                            inputs.add(arg)
                
                # Total tensors involved
                all_tensors = inputs | outputs
                if len(all_tensors) > self.max_tensors:
                    can_add = False
            
            if can_add:
                if current_shape is None:
                    current_shape = shape
                current_subgraph_nodes.append(node)
            else:
                if current_subgraph_nodes:
                    self.subgraphs.append(current_subgraph_nodes)
                current_subgraph_nodes = [node]
                current_shape = shape
        
        if current_subgraph_nodes:
            self.subgraphs.append(current_subgraph_nodes)
            
        return self.subgraphs
