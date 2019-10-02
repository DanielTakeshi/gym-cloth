# Gym Cloth

This creates a gym environment based on our cloth simulator. The path directory
is structured following [standard gym conventions][1], and we also include our
`.pyx` files here for Cython compilation.

Platforms tested:

- Mac OS X (renderer working)
- Ubuntu 16.04 (renderer not working, unfortunately)
- Ubuntu 18.04 (renderer working)

Python versions tested:

- Python 2.7
- Python 3.6

The code is hopefully agnostic to Python 2 or 3, however we strongly recommend
using Python 3. We have not done any serious Python 2 testing for many months.


## Installation and Code Usage

1. Make a new virtualenv. For example, if you're using Python 2 and you put
your environments in a directory `~/Envs`:

   ```
   virtualenv --python=python2 ~/Envs/py2-clothsim
   ```

   If you're not working with the physical robots, don't use the
   `--system-site-packages` option yet. We have ROS installed on our machines,
   and I think if we use the physical robots, we want that option.

2. Run `pip install -r requirements.txt`. This should work regardless of
whether you're using Python 2 or Python 3.

3. Run `python setup.py install`. This should automatically "cythonize" the
Python `.pyx` files. An alternative is to do `python setup.py develop` in
development mode. This has the advantage in that code changes in
package-dependent files will automatically be updated when you run code, and
you avoid having to "re-install" the package. However, since we use Cython code
in files that end with `.pyx`, those have to be re-compiled each time we run
the code. Thus, they automatically require another `python setup.py` call
anyway, so it seems like the distinction between install mode and develop mode
doesn't matter here. For example, if we want to run a fictitious `demo.py`
script, we can do this to test:

   ```
   python setup.py install ; python examples/demo.py
   ```

   So far this setup is working fine for us that we haven't really seen a need
   to change things around.

See `examples` for usage, and how to save videos. Simply run with `python
examples/[script_name].py`.  You will need this package installed (see above)
so that you can use commands like `from gym_cloth import ...` regardless of
your current directory.



## Renderer Installation

These instructions have been tested on Mac OS X and Ubuntu 18.04. For some
reason, we have not been able to get this working for Ubuntu 16.04. Furthermore,
for Ubuntu 18.04, you might need sudo access for `make -j4 install`.

Currently the simulation is rendered in an independent C++ program. To set up
the renderer,

1. Navigate to `render/ext/libzmq`. Run
```
mkdir build
cd build
cmake ..
make -j4 install
```
2. Navigate to `render/ext/cppzmq`. Again run
```
mkdir build
cd build
cmake ..
make -j4 install
```
3. Navigate to `render`. Run
```
mkdir build
cd build
cmake ..
make
```

Finally you should have an executable `clothsim` in `render/build`. **To test
that it is working, go to `render/build` and run `./clothsim` on the command
line. You should see an empty window appear. There should be no segmentation
faults.** Occasionally I have seen it fail on installed machines, but normally
rebooting fixes it.

The renderer will have to be run in an independent process. See
`examples/demo.py` for how we can do this within Python.

Notes:

- If you make changes to `width`, `height`, or `render_port` in
  `cfg/env_config.yaml`, please also update `num_width_points`,
  `num_height_points`, and `render_port` respectively in
  `render/scene/pinned2.json`.

- It's easier to change the viewing angle by directly adjusting values in
  `clothSimulator.cpp`, rather than with the mouse and GUI. When you adjust the
  camera angles, be sure to re-compile the renderer using the instructions
  above. You only need to re-compile `render`, not the other two.

[1]:https://github.com/openai/gym/tree/master/gym/envs
[2]:https://github.com/openai/gym/pull/1314
