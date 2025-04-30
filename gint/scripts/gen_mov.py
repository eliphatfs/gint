import rich
import pyperclip


template = """
class MovF{dst}F{src}(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf{dst}] = list(state[ispec.rf{src}])

"""

epi = """
all_moves = [
%s
]
"""

def main():
    s = ""
    e = []
    for src in range(4):
        for dst in range(4):
            if src != dst:
                s += template.format(dst=dst, src=src)
                e.append('    MovF{dst}F{src}(),'.format(dst=dst, src=src))
    s += epi % '\n'.join(e)
    pyperclip.copy(s)
    rich.print(s.replace("[", "\\["))


if __name__ == '__main__':
    main()
