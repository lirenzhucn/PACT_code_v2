#!/usr/bin/env python3
"""Combined setup script for all the extensions"""

from distutils.core import setup, Extension
import numpy

# define the extension module
RING_PACT_SPEEDUP_MODULE =\
    Extension('ring_pact_speedup',
              sources=['ring_pact_speedup.c'],
              depends=['ring_pact_speedup.h'],
              include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[RING_PACT_SPEEDUP_MODULE])
