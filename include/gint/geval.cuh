#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include "opcode.h"


__global__ void geval(char* text, char** data)
{
    int pc = 0;
    __shared__ float gpr[8 * 32];
    while (true)
    {
        int32_t insn = *(int32_t*)(text + pc);
        int8_t* insn_exp = (int8_t*)&insn;
        pc += 4;
        switch ((EOpCode)insn_exp[0]) {
            case EOpCode::HALT:
                return;
            case EOpCode::ADD_F32:
                gpr[insn_exp[1]] = gpr[insn_exp[2]] + gpr[insn_exp[3]];
                break;
            case EOpCode::MUL_F32:
                gpr[insn_exp[1]] = gpr[insn_exp[2]] * gpr[insn_exp[3]];
                break;
            case EOpCode::DIV_F32:
                gpr[insn_exp[1]] = gpr[insn_exp[2]] / gpr[insn_exp[3]];
                break;
            case EOpCode::SUB_F32:
                gpr[insn_exp[1]] = gpr[insn_exp[2]] - gpr[insn_exp[3]];
                break;
            case EOpCode::LD_32:
                gpr[insn_exp[1]] = *(float*)data[insn_exp[2]];
                break;
            case EOpCode::ST_32:
                *(float*)data[insn_exp[2]] = gpr[insn_exp[1]];
                break;
            default:
                __builtin_unreachable();
                break;
        }
    }
}
