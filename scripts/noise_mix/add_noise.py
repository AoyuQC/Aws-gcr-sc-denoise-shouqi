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


def PostProcess(clean_path, dirName, trainPath, noise_dist, pure_noise_path):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + \
                PostProcess(clean_path, fullPath, trainPath, noise_dist, pure_noise_path)
        else:
            if fullPath.split('_')[-1] == 'noise.wav':
                print(fullPath)
                valid_name = '_'.join(fullPath.split(
                    '/')[-1].split('_')[0:-1])+'_'+fullPath.split('/')[-2]+'.wav'
                valid_output = osp.join(
                    *fullPath.split('/')[0:-3], pure_noise_path, valid_name)
                shutil.copyfile(fullPath, valid_output)
                print(valid_output)
            elif fullPath.split('.')[-1] == 'wav':
                allFiles.append(fullPath)
                valid_name = fullPath.split(
                    '/')[-1].split('.')[0]+'_'+fullPath.split('/')[-2]+'.wav'
                valid_output = osp.join(
                    *fullPath.split('/')[0:-3], noise_dist, valid_name)
                # print(fullPath)
                # print(valid_output)
                shutil.copyfile(fullPath, valid_output)
                rawPath = osp.join(clean_path, fullPath.split('/')[-1])
                shutil.copyfile(rawPath, osp.join(trainPath, valid_name))

    return allFiles


def ProcessAudio(dd):
    for ds in glob.glob(osp.join(dd, '*.wav')):
        print(ds)
        song_raw = AudioSegment.from_wav(ds)
        song_16k = song_raw.set_frame_rate(16000)
        song = song_16k.set_channels(1)
        song.export(ds, format='wav')


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
        '--dir', help='directory for segan data', required=True
    )
    parser.add_argument(
        '--snr', help='signal noise ratio, db', required=True, nargs='+', type=int
    )
    parser.add_argument(
        '--level', help='level method for speech', required=True, default='P.56'
    )
    args = parser.parse_args()
    # Make sure this is reproducible
    np.random.seed(42)

    d = Dataset(args.level)
    dd = args.dir

    ProcessAudio(osp.join(dd, 'clean_trainset_wav_16k_raw'))
    ProcessAudio(osp.join(dd, 'pure_noise'))

    # All files can be added one by one or by folder. Adding a folder will add
    # all speech files inside that folder recursively if recursive=True.
    d.add_speech_files(
        osp.join(dd, 'clean_trainset_wav_16k_raw'), recursive=True)

    # When adding noises, you can give a "nickname" to each noise file. If you do not
    # give it a name, the name will be the file name without the '.wav' extension
    # for i in range(21):
    #     if i == 5:
    #         continue
    #     d.add_noise_files(
    #         '/Users/aoyuzhan/Workplace/Datalab/Datalab_3_ShouqiNLP/scripts/noise_clip/'+str(i+1)+'.wav', name='noise'+str(i))
    for ddi in glob.glob(osp.join(dd, 'pure_noise', '*.wav')):
        d.add_noise_files(ddi, name=ddi.split('/')[-1].split('.')[0])
    # d.add_noise_files(
    #     '/Users/aoyuzhan/Workplace/Datalab/Datalab_3_ShouqiNLP/scripts/noise_clip/119_1_clip_0_0.wav', name='noise119_1')
    # d.add_noise_files(
  #     '/Users/aoyuzhan/Workplace/Datalab/Datalab_3_ShouqiNLP/scripts/noise_clip/119_26_clip_0_0.wav', name='noise119_26')
    # d.add_noise_files(
    #     '/Users/aoyuzhan/Workplace/Datalab/Datalab_3_ShouqiNLP/scripts/noise_clip/119_61_clip_0_0.wav', name='noise119_61')
    # d.add_noise_files(
    #     '/Users/aoyuzhan/Workplace/Datalab/Datalab_3_ShouqiNLP/scripts/noise_clip/67_3_clip_0_0.wav', name='noise67_3')
    # d.add_noise_files('/home/jfsantos/data/multichannel_noises/restaurant_ch01.wav', name='restaurant')
    # d.add_noise_files('/home/jfsantos/data/multichannel_noises/cafeteria_ch01.wav', name='cafeteria')
    # d.add_noise_files('/home/jfsantos/data/multichannel_noises/traffic_ch01.wav', name='traffic')

    # Adding reverb files works like adding noise files
    # d.add_reverb_files('/home/jfsantos/data/RIR_sim/rir_0.2_1.wav')
    # d.add_reverb_files('/home/jfsantos/data/RIR_sim/rir_0.8_1.wav')

    # When generating a dataset, you can choose which SNRs will be used and how many
    # files per condition you want to be generated.
    ClearOutputDir(osp.join(dd, 'noise_trainset_wav_16k_raw'))
    ClearOutputDir(osp.join(dd, 'clean_trainset_wav_16k'))
    ClearOutputDir(osp.join(dd, 'noise_trainset_wav_16k'))
    ClearOutputDir(osp.join(dd, 'pure_noise_trainset_wav_16k'))
    # prepare dataset for mix application
    tt_path = ['tr', 'cv', 'tt']
    ss_path = ['mix', 's1', 's2']
    for data_type in tt_path:
        for speaker in ss_path:
            ClearOutputDir(osp.join(dd, data_type, speaker))

    # d.generate_dataset([-3,0,3], osp.join(dd,'noise_trainset_wav_16k_raw'), files_per_condition=5)
    d.generate_dataset(args.snr, osp.join(
        dd, 'noise_trainset_wav_16k_raw'), files_per_condition=len(args.snr))
    # d.generate_dataset([-6, -3, 0, 3, 6], '/tmp/noise_plus_reverb_dataset', files_per_condition=5)

    # post process for data training
    clean_path = osp.join(dd, 'clean_trainset_wav_16k_raw')
    top_path = osp.join(dd, 'noise_trainset_wav_16k_raw')
    train_path = osp.join(dd, 'clean_trainset_wav_16k')
    noise_path = osp.join(dd, 'noise_trainset_wav_16k')
    pure_noise_path = osp.join(dd, 'pure_noise_trainset_wav_16k')
    PostProcess(clean_path, top_path, train_path, noise_path, pure_noise_path)

    # mix_path = osp.join(dd, 'mix')
    # s1_path = osp.join(dd, 's1')
    # s2_path = osp.join(dd, 's2')
    # shutil.copytree(noise_path, mix_path, dirs_exist_ok=True)
    # shutil.copytree(train_path, s1_path, dirs_exist_ok=True)
    # shutil.copytree(pure_noise_path, s2_path, dirs_exist_ok=True)
    raw_list = os.listdir(noise_path)
    shuffle(raw_list)

    train_ratio = 0.8
    val_ratio = 0.1

    train, val, test = np.split(raw_list, [int(train_ratio*len(raw_list)), int((train_ratio+val_ratio)*len(raw_list))])

    dirlist = []
    dirlist.append(train)
    dirlist.append(val)
    dirlist.append(test)

    for idx, ddir in enumerate(dirlist):
        for ff_name in ddir:
            shutil.copyfile(osp.join(noise_path,ff_name), osp.join(dd,tt_path[idx],'mix',ff_name))
            shutil.copyfile(osp.join(train_path,ff_name), osp.join(dd,tt_path[idx],'s1',ff_name))
            shutil.copyfile(osp.join(pure_noise_path,ff_name), osp.join(dd,tt_path[idx],'s2',ff_name))

if __name__ == '__main__':
    main()
