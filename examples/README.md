# Examples

Example code usage. If you want to test something, make a new script and
document it below. Please also add a new configuration file. Here are files,
roughly in chronological order, so the later files are more likely to have up
to date settings.

- `demo_spaces.py`: main test bed that I have been using for tuning RL and to
  debug before using the analytic script `analytic.py`. **This goes with
  `cfg/demo_spaces.yaml`**.

- `analytic.py`: use for testing our analytic policies, to serve as baselines
  with the RL algorithms. **This goes with `cfg/demo_baselines.yaml`**, which
  is the actual one that we have been using.

- `env_init.py`: use to test the sampling of initial environment states for
  various "tiers" of difficulty level.

Older and deprecated scripts but might keep these around:

- `demo_bed.py`: the initial bed-making test with hard-coded actions. This is
  the video Professor Goldberg used for his BAIR retreat talk.

- `demo_render.py`: builds on top of `demo_spaces.py` and demonstrates pyrender
  and blender rendering capabilities.


# Creating Videos

Recording videos of the simulator render is quite simple on Mac OS X. To screen
record simply do the following:

1. Open the QuickTime Player app (should be natively installed)
2. Click File > New Screen Recording and then click the red dot.
3. Drag your mouse to cover the renderer window.
4. Let go to begin recording.
5. Press the "Stop" icon in the Menu Bar (top of the screen) to stop.
6. The video should come up in a new window. Close the window to save the file.
7. You can easily trim the beginning and ending of videos in QuickTime.

To increase the speed (e.g., to 2x speed) one possibility is to use iMovie.
