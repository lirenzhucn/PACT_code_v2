#!/usr/bin/env python3
"""Combined setup script for all the extensions"""

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import numpy


class build_ext_openmp(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        if compiler == 'msvc':
            for e in self.extensions:
                e.extra_compile_args = ['/openmp']
        elif compiler == 'gcc':
            for e in self.extensions:
                e.extra_compile_args = ['-fopenmp']
                e.extra_link_args = ['-lgomp']
        super().build_extensions()

# define the extension module
RING_PACT_SPEEDUP_MODULE =\
    Extension('ring_pact_speedup',
              sources=['ring_pact_speedup.c'],
              depends=['ring_pact_speedup.h'],
              include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[RING_PACT_SPEEDUP_MODULE],
      cmdclass={'build_ext': build_ext_openmp})
