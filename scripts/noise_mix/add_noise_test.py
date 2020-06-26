#!/usr/bin/env python

from maracas.dataset import Dataset
import argparse
import numpy as np
import os
from os import path as osp
import shutil
import glob
from pydub import AudioSegment
import sys
from random import shuffle

def ClearOutputDir(test_path):
    if osp.exists(test_path):
        # output dir exists
        if len(os.listdir(test_path)) != 0:
            print('Ouput dir {} exists and not empty '.format(test_path))
            # TODO: show the description of directory
            c = input('Clear?: y/n ')
            if c == 'y':
                print('Erasing...')
                shutil.rmtree(test_path)
                os.makedirs(test_path)
            else:
                print('Please check the contents of your output directory')
                sys.exit(1)
    else:
        # make new output dir
        os.makedirs(test_path)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_file', help='input file path for preprocess', required=True
    )
    args = parser.parse_args()

if __name__ == '__main__':
    main()
