import argparse
from gint.kernel.interpreter.main import build_interpreter_main_nvptx


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-o", "--output-path", type=str, required=True)
    args = argp.parse_args()
    
    mod = build_interpreter_main_nvptx()
    with open(args.output_path, "w") as fo:
        fo.write(str(mod))
