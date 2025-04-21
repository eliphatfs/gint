#pragma once
#include <stdint.h>

enum class EOpCode : int16_t {
    ADD_F32,
    MUL_F32,
    DIV_F32,
    SUB_F32,
    LD_32,
    ST_32,
    HALT
};
