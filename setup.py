# distutils

# Compile with: `python setup.py build_ext --inplace`
#
# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
#
# setup(name='gym_cloth',
#     version='0.0.1',
#     packages=['gym_cloth',])
#
# BUT ... I do not know how to use this in the context of a _package_, so that I
# can import it like a normal pip package. That's why I use setuptools.


# setuptools
# ------------------------------------------------------------------------------
# Useful references:
# ------------------------------------------------------------------------------
# https://cython.readthedocs.io/en/latest/src/quickstart/build.html
# https://setuptools.readthedocs.io/en/latest/setuptools.html?highlight=cython
# https://github.com/Technologicat/setup-template-cython
# http://docs.cython.org/en/latest/src/tutorial/external.html
# https://github.com/pypa/sampleproject
# ------------------------------------------------------------------------------
# I *think* this one is working ... and if we add more the pattern is obvious.
# Most references say to add a `build_ext` command, but that's with distutils,
# as in `python setup.py build_ext --inplace`. But, I think with setuptools the
# `python setup.py install` will automatically build the source files. I can
# only tell by looking at the compilation output. But we have to do this each
# time we change the code.
# ------------------------------------------------------------------------------

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension('gym_cloth.physics.cloth',   ['gym_cloth/physics/cloth.pyx']   ),
    Extension('gym_cloth.physics.gripper', ['gym_cloth/physics/gripper.pyx'] ),
    Extension('gym_cloth.physics.point',   ['gym_cloth/physics/point.pyx']   ),
]

setup(
    name='gym_cloth',
    version='0.0.1',
    description='Basic cloth simulator for reinforcement learning',
    author='Ryan Hoque, Daniel Seita',
    packages=['gym_cloth', 'gym_cloth.envs', 'gym_cloth.blender', 'gym_cloth.physics'],
    package_data={'gym_cloth': ['blender/*.obj']},
    ext_modules=cythonize(ext_modules),
    zip_safe=False,
    annotate=True,
)
