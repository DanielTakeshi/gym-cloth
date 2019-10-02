"""Random utilities.
"""
import sys


def write_to_file(cloth, filename):
    """Write cloth's state to file.
    """
    f = open(filename, "w+")
    pickle.dump(cloth, f)
    f.close()


def load_from_file(filename):
    """Load a past state from file.
    """
    f=open(filename,'rb')
    try:
        return pickle.load(f)
    except EOFError:
        print('Nothing written to file.')
