#!/usr/bin/env blender --python
import bpy
import sys
import argparse

if __name__ == "__main__":
    # This will work, but a bit clumsy. Simply run with:
    #
    #   blender --python blender_render/test.py -- filepath
    #
    # Where we use the arguments after the double dash.

    argv = sys.argv
    argv = argv[argv.index('--')+1:]
    print(argv)
    filepath = argv[0]
    print('loading from: {}'.format(filepath))

    #Delete the starting cube
    bpy.ops.object.delete(use_global=False)

    print('\n\nsuccessfully started blender\n\n')

    # The reason the above is needed is becaues argparse and blender have their
    # own arguments, and passing in args for both results in ambiguity.
    # https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script

    #parser = argparse.ArgumentParser()
    #parser.add_argument('-path', type=str)
    #args = parser.parse_args()
    #print(args.path)
