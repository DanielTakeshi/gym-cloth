# Gym Cloth

Quick logistics overview: this is *one* of the code bases used in our paper "Deep Imitation Learning of Sequential Fabric Smoothing From an Algorithmic Supervisor" with [arXiv here][3] and [project website here][4]. The arXiv version will have the most up-to-date version of the paper. If you find the code or other related resources useful, please consider citing the paper:

```
@inproceedings{seita_fabrics_2020,
    author = {Daniel Seita and Aditya Ganapathi and Ryan Hoque and Minho Hwang and Edward Cen and Ajay Kumar Tanwani and Ashwin Balakrishna and Brijen Thananjeyan and Jeffrey Ichnowski and Nawid Jamali and Katsu Yamane and Soshi Iba and John Canny and Ken Goldberg},
    title = {{Deep Imitation Learning of Sequential Fabric Smoothing From an Algorithmic Supervisor}},
    booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    Year = {2020}
}
```

<hr>

This creates a gym environment based on our cloth simulator. The path directory
is structured following [standard gym conventions][1], and we also include our
`.pyx` files here for Cython compilation.

Platforms tested:

- Mac OS X (renderer working)
- Ubuntu 16.04 (renderer not working, unfortunately)
- Ubuntu 18.04 (renderer working)

Please use **Python 3.6**.

## Installation and Code Usage

1. Make a new virtualenv or conda env. For example, if you're using conda envs,
   run this to make and then activate the environment:

   ```
   conda create -n py3-cloth python=3.6
   conda activate py3-cloth
   ```

2. Run `pip install -r requirements.txt` to install dependencies.

3. Run `python setup.py install`. This should automatically "cythonize" the
   Python `.pyx` files. An alternative is to do `python setup.py develop` in
   development mode. This has the advantage in that code changes in
   package-dependent files will automatically be updated when you run code, and
   you avoid having to "re-install" the package. However, since we use Cython
   code in files that end with `.pyx`, those have to be re-compiled each time
   we run the code. Thus, they automatically require another `python setup.py`
   call anyway, so it seems like the distinction between install mode and
   develop mode doesn't matter here. For example, if running a fictitious
   `demo.py` script, I usually do this each time I run code:

   ```
   python setup.py install ; python examples/<script_name>.py
   ```

   So far this setup is working fine for us that we haven't really seen a need
   to change things around.

For quick testing, try running the policies using the provided
`examples/analytic.py` script. This is the main script that we use to generate
demonstration data for experiments. For example, this should work right away:

```
python examples/analytic.py oracle --max_episodes=400 --seed=1336 --tier=1
```

To actually *visualize* the renderer, you need to install it, *and* change the
appropriate config file in `cfg/` so that the `render_opengl` setting is
`True`.


## Renderer Installation

These instructions have been tested on Mac OS X and Ubuntu 18.04. For some
reason, we have not been able to get this working for Ubuntu 16.04.  *For
Ubuntu 18.04, you might need sudo access for `make -j4 install`.* Currently the
simulation is rendered in an independent C++ program. To set up the renderer,

1. Navigate to `render/ext/libzmq`. Run
```
mkdir build; cd build
cmake ..
make -j4 install
```

2. Navigate to `render/ext/cppzmq`. Again run
```
mkdir build; cd build
cmake ..
make -j4 install
```
(This step may not actually work.)

3. Navigate to `render`. Run
```
mkdir build; cd build
cmake ..
make
```

Finally you should have an executable `clothsim` in `render/build`. **To test
that it is working, go to `render/build` and run `./clothsim` on the command
line. You should see an empty window appear. There should be no segmentation
faults.** Occasionally I have seen it fail on installed machines, but normally
rebooting fixes it.

Notes:

- If you make changes to `width`, `height`, or `render_port` in
  `cfg/env_config.yaml`, please also update `num_width_points`,
  `num_height_points`, and `render_port` respectively in
  `render/scene/pinned2.json`.

- It's easier to change the viewing angle by directly adjusting values in
  `clothSimulator.cpp`, rather than with the mouse and GUI. When you adjust the
  camera angles, be sure to re-compile the renderer using the instructions
  above. You only need to re-compile `render`, not the other two.

- Note (Updated December 2020): with Ubuntu 18.04 and a conda environment,
  running `./clothsim` seems to result in a frozen screen. Also, the second
  step of the installation above seems to fail (though [this fix may help][5]),
  but nonetheless, running it with cloth in it will seem to work normally.

[1]:https://github.com/openai/gym/tree/master/gym/envs
[2]:https://github.com/openai/gym/pull/1314
[3]:https://arxiv.org/abs/1910.04854
[4]:https://sites.google.com/view/fabric-smoothing
[5]:https://github.com/zeromq/cppzmq/issues/334
