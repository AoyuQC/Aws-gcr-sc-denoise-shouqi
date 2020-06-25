#!/usr/bin/env python

import argparse
from pydub import AudioSegment
import glob
import os
from os import path as osp
import shutil
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def ProcessAudio(args, fullPath):
    #start_min = args.clip_start_min
    #start_sec = args.clip_start_sec
    #end_min = args.clip_end_min
    #end_sec = args.clip_end_sec
    #start_time = start_min*60+start_sec
    #end_time = end_min*60+end_sec
    save_path = args.output_dir

    # ffmpeg_extract_subclip("test.mp4", start_time, end_time, targetname="test_clip.mp4")
    # make output dir
    out_dir = osp.join(save_path, *fullPath.split('/')[1:-1])
    out_name = fullPath.split('/')[-1].split('.')[0]
    up_name = fullPath.split('/')[-2]
    if osp.exists(out_dir) == False:
        os.makedirs(out_dir)

    # # clip_time = end_time-start_time
    print("to tranfer {}".format(fullPath))
    try:
        song_raw = AudioSegment.from_mp3(fullPath)
        song_16k = song_raw.set_frame_rate(16000)
        song = song_16k.set_channels(1)
        mill = 1000
        dur = song_raw.duration_seconds-1
        seg_sec = dur * mill
        whole = int(song_raw.duration_seconds/dur)
        for i in range(whole):
            start = i * seg_sec
            end = start+seg_sec
            temp = song[start:end]
            out_name_temp = osp.join(out_dir, up_name+'_'+out_name +
                                     '_clip_'+str(i)+'.wav')
            temp.export(out_name_temp, format="wav")
            print("finish transfer {}".format(out_name_temp))
    except:
        print("{} fail!!!!".format(fullPath))
        pass


def TransferListOfFiles(args, dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName) 
    allFiles = list()
       # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + TransferListOfFiles(args, fullPath)
        else:
            allFiles.append(fullPath)
            # create sub clip voice
            ProcessAudio(args, fullPath)

    return allFiles

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

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--batch', help='whether to batch process or not', required=True, type=boolean_string, default=False
    )
    parser.add_argument(
        '--start_time_list', nargs='+', help='the start time list of clip part', type=float)
    parser.add_argument(
        '--end_time_list', nargs='+', help='the end time list of clip part', type=float)
    # parser.add_argument(
    #    '--clip_start_min', help='the start minute of clip part', type=float)
    # parser.add_argument(
    #    '--clip_start_sec', help='the start second of clip part', type=float)
    # parser.add_argument(
    #    '--clip_end_min', help='the end minute of clip part', type=float)
    # parser.add_argument(
    #    '--clip_end_sec', help='the end second of clip part', type=float)
    # parser.add_argument(
    #    '--inter_sec', help='the interval second of clip part', type=float)
    # parser.add_argument(
    #    '--dur_sec', help='the duration second of each clip part', type=float)
    parser.add_argument(
        '--output_dir', help='output save path for clip audios')
    parser.add_argument(
        '--input_dir', help='input path for raw audios')
    parser.add_argument(
        '--input_file', help='input file path for non-batch mode')
    parser.add_argument(
        '--output_single_dir', help='output save path for non-batch mode')
    parser.add_argument(
        '--noise_dur', help='duration of noise', type=float)
    args = parser.parse_args()

    batch_mode = args.batch
    print(batch_mode)

    if batch_mode == True:
        test_path = args.output_dir
    else:
        print(args.output_single_dir)
        test_path = args.output_single_dir
    
    print(test_path)

    # clear output dir
    if batch_mode == True:
        ClearOutputDir(test_path)
 
    if batch_mode == True:
        TransferListOfFiles(args, args.input_dir)
    else:
        input_file = args.input_file
        start_time_list = args.start_time_list
        end_time_list = args.end_time_list

        if len(start_time_list) != len(end_time_list):
            raise AssertionError("length of start_time_list {} not equal to length of end_time_list {}".format(
                len(start_time_list), len(end_time_list)))
        
        for cnt, ele in enumerate(start_time_list):
            start_time = ele
            end_time = end_time_list[cnt]
            print(start_time)
            print(end_time)
            out_name = osp.join(test_path,input_file.split('/')[-1].split('.')[0]+'_'+str(cnt)+'.wav')
            ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=out_name)
            # change length to desired length
            cnt = int(args.noise_dur/(end_time-start_time))+1
            wavs = [AudioSegment.from_wav(out_name) for i in range(cnt)]
            combined = wavs[0]
            for wav in wavs[1:]:
                combined = combined.append(wav)
            combined.export(out_name, format="wav")
    # # clip_time = end_time-start_time
    # song_raw = AudioSegment.from_wav("439.wav")
    # song_16k = song_raw.set_frame_rate(16000)
    # song = song_16k.set_channels(1)
    # mill = 1000
    # dur = song_raw.duration_seconds-1
    # seg_sec = dur * mill
    # whole = int(song_raw.duration_seconds/dur)
    # for i in range(whole):
    #     start = i * seg_sec
    #     end = start+seg_sec
    #     temp = song[start:end]
    #     temp.export('./slice/slice_'+str(i)+'.wav',format="wav")
    #     print(temp.channels)


if __name__ == '__main__':
    main()
