import numpy as np
from typing import List, Dict, Any
from ..kernel.interpreter.main import (
    INSNS, Halt, LoadImm, FAddImm, FMulImm, FMAImm, 
    LoadGlobalF32, StoreGlobalF32, LoadGlobalF16, StoreGlobalF16,
    LoadGlobalBF16, StoreGlobalBF16, LoadGlobalU8, FApprox,
    LoadImm4F, LoadImm4I
)

def pprint_bytecode(bc: List[List[int]]) -> str:
    """
    Returns a human-readable string representation of gint bytecode.
    """
    # Reverse map INSNS
    id_to_name = {opid: Insn.__name__ for Insn, opid in INSNS.items()}
    
    lines = []
    
    # Define instructions with special argument decoding
    float_imms = {INSNS[LoadImm], INSNS[FAddImm], INSNS[FMulImm], INSNS[FApprox]}
    global_io = {
        INSNS[LoadGlobalF32], INSNS[StoreGlobalF32], 
        INSNS[LoadGlobalF16], INSNS[StoreGlobalF16],
        INSNS[LoadGlobalBF16], INSNS[StoreGlobalBF16],
        INSNS[LoadGlobalU8]
    }
    fma_imm = INSNS[FMAImm]
    packed_imm = {INSNS[LoadImm4F], INSNS[LoadImm4I]}

    for pc, (opid, imm) in enumerate(bc):
        name = id_to_name.get(opid, f"UNKNOWN({opid})")
        
        arg_str = ""
        if opid in float_imms:
            val = np.int32(imm).view(np.float32).item()
            arg_str = f"{val:.6f}"
        elif opid in global_io:
            offset = imm // 16
            arg_i = imm % 16
            arg_str = f"offset={offset}, arg_i={arg_i}"
        elif opid == fma_imm:
            # FMAImm stores [mul, add] as float16 packed into one int32
            packed = np.array([imm], dtype=np.int32).view(np.float16)
            mul_val = packed[0].item()
            add_val = packed[1].item()
            arg_str = f"mul={mul_val:.4f}, add={add_val:.4f}"
        elif opid in packed_imm:
            arg_str = f"0x{imm:08x}"
        elif imm != 0:
            arg_str = str(imm)
            
        lines.append(f"{pc:04d}: {name:<20} {arg_str}")
        
        if opid == INSNS[Halt]:
            break
            
    return "\n".join(lines)

def dump_bytecode(bc: List[List[int]], filename: str):
    """
    Pretty-prints and dumps bytecode to a file.
    """
    representation = pprint_bytecode(bc)
    with open(filename, 'w') as f:
        f.write(representation)
        f.write("\n")
