import unittest
import numpy

from gint import analyze_bytecode, BytecodeStats
from gint.kernel.interpreter.main import INSNS, VARIANTS
from gint.host.analyzer import known_opcodes
from gint.host.executor import select_variant


class TestAnalyzerCoverage(unittest.TestCase):

    def test_every_opcode_has_stack_effect(self):
        """The hand-maintained _EFFECTS table must cover every registered opcode."""
        registered = set(INSNS.values())
        known = known_opcodes()
        missing = registered - known
        self.assertFalse(missing, f"opcodes without stack-effect entry: {missing}")


class TestAnalyzerBasics(unittest.TestCase):

    def test_empty_bytecode(self):
        s = analyze_bytecode([[0, 0]])
        self.assertEqual(s.num_instructions, 0)
        self.assertEqual(s.max_stack, 0)
        self.assertEqual(s.max_reg_idx, -1)

    def test_simple_add(self):
        # fldg_1d, fldg_1d, fadd, fstg_1d, halt
        bc = [[15, 0], [15, 0], [2, 0], [16, 0], [0, 0]]
        s = analyze_bytecode(bc)
        self.assertEqual(s.max_stack, 2)
        self.assertEqual(s.max_reg_idx, -1)

    def test_register_use(self):
        # fldg_1d, fstore_reg2, fload_reg2, fstg_1d, halt
        bc = [[15, 0], [97, 0], [89, 0], [16, 0], [0, 0]]
        s = analyze_bytecode(bc)
        self.assertEqual(s.max_reg_idx, 2)
        self.assertEqual(s.regs_used, frozenset({2}))

    def test_numpy_array_input(self):
        bc = numpy.array([[15, 0], [15, 0], [2, 0], [0, 0]], dtype=numpy.int32)
        s = analyze_bytecode(bc)
        self.assertEqual(s.num_instructions, 3)
        # Also accepts flat layout
        s2 = analyze_bytecode(bc.reshape(-1))
        self.assertEqual(s2, s)


class TestVariantSelection(unittest.TestCase):

    def test_pointwise_picks_small(self):
        # depth ≤ 3, no regs → should fit s7
        bc = [[15, 0], [15, 0], [2, 0], [16, 0], [0, 0]]
        self.assertEqual(select_variant(bc), 's7')

    def test_high_register_forces_large(self):
        # uses reg 7 (FLoadReg7=94 / FStoreReg7=102) → only l12 has 8 regs
        bc = [[15, 0], [102, 0], [94, 0], [16, 0], [0, 0]]
        self.assertEqual(select_variant(bc), 'l12')

    def test_low_register_keeps_small(self):
        # uses reg 3 (FLoadReg3=90 / FStoreReg3=98) — fits in s7's regs=4
        bc = [[15, 0], [98, 0], [90, 0], [16, 0], [0, 0]]
        self.assertEqual(select_variant(bc), 's7')

    def test_deep_stack_forces_large(self):
        # Push 8 values then pop them; s7's max_stack=7 → should not fit
        bc = [[11, 0]] * 8 + [[22, 0]] * 8 + [[0, 0]]
        self.assertEqual(select_variant(bc), 'l12')


class TestVariantsTable(unittest.TestCase):

    def test_variant_layout(self):
        # Sanity check on the configured variants — small must be strictly
        # smaller than large in pool size, otherwise selection is ambiguous.
        s_pool, s_regs, s_stack = VARIANTS['s7']
        l_pool, l_regs, l_stack = VARIANTS['l12']
        self.assertLess(s_pool, l_pool)
        self.assertLessEqual(s_regs, l_regs)
        self.assertLessEqual(s_stack, l_stack)


if __name__ == '__main__':
    unittest.main()
