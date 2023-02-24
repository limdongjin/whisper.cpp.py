import os, sys
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
import numpy

CFLAGS = { 
    'darwin': '-DGGML_USE_ACCELERATE -O3 -std=gnu11',   # -fPIC -pthread -mf16c -mfma -mavx -mavx2',
    'default': '-mavx -mavx2 -mfma -mf16c -O3 -std=gnu11 -fPIC -pthread'
}
CXXFLAGS = {
    'darwin': '-DGGML_USE_ACCELERATE -O3 -std=c++11',
    'default': '-mavx -mavx2 -mfma -mf16c -O3 -std=c++11 -fPIC -pthread'
}
LDFLAGS = {
    'darwin': '-framework Accelerate'
}

platform = sys.platform
os.environ['CFLAGS'] = CFLAGS.get(platform, CFLAGS['default'])
os.environ['CXXFLAGS'] = CXXFLAGS.get(platform, CXXFLAGS['default'])
if platform == 'darwin':
    os.environ['LDFLAGS'] = LDFLAGS['darwin']

ext_modules = [
    Extension(
        name="whisper_cpp_py",
        sources=["whisper_cpp_py.pyx", "whisper.cpp/whisper.cpp"],
        language="c++",
        extra_compile_args=["-std=c++11"], 
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

ext_modules = cythonize(ext_modules)

whisper_cpp_ggml = ('whisper_cpp_ggml', {'sources': ['whisper.cpp/ggml.c']})

setup(
    name='whisper.cpp.py',
    version='0.1',
    description='whisper_cpp_py',
    author='limdongjin',
    author_email='geniuslim27@gmail.com',
    libraries=[whisper_cpp_ggml],
    ext_modules = cythonize("whisper_cpp_py.pyx"),
    include_dirs = ['./whisper.cpp/', numpy.get_include()],
    install_requires=[
        'numpy'
    ],
)
