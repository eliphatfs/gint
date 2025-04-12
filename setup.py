import os
os.environ["NVCC_APPEND_FLAGS"] = "--allow-unsupported-compiler"
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


with open("gint/version.py", "r") as fh:
    exec(fh.read())
    __version__: str


_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gint", "csrc")

nvcc_flags = [
    '-O3', '-use_fast_math', '-std=c++17'
]

if os.name == "posix":
    c_flags = ['-O3', '-std=c++17']
elif os.name == "nt":
    c_flags = ['/O2']

    # find cl.exe
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(
                r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition),
                reverse=True)
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

setup(
    name='gint',  # package name, import this to use python API
    version=__version__,
    ext_modules=[
        CUDAExtension(
            name='gint._C',  # extension name, import this to use CUDA API
            sources=[os.path.join(_src_path, f) for f in [
                'binding.cpp',
                'plugin.cu',
            ]],
            include_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)
