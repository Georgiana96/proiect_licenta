import pandas
import subprocess
import os
import glob
from subprocess import check_output
from preprocess import constants


class WavGenerator:

    @staticmethod
    def get_file_duration(file_path):
        a = str(check_output('ffprobe -i  "' + file_path + '" 2>&1 |findstr "Duration"', shell=True))
        a = a.split(",")[0].split("Duration:")[1].strip()
        h, m, s = a.split(':')
        duration = int(h) * 3600 + int(m) * 60 + float(s)
        return int(duration)

    @staticmethod
    def split_file(mp4_file_path, file_duration, wav_file_path):
        default_duration = constants.DEFAULT_DURATION
        start_even = 0
        i = 0
        while start_even + default_duration <= file_duration:
            command = "ffmpeg -ss " + str(start_even) + " -i " + mp4_file_path + " -t " + str(
                default_duration)
            command += " -ab 160k -ac 1 -ar 44100 -vn " + wav_file_path + '_' + str(i) + '.wav'
            subprocess.call(command, shell=True)
            i += 1
            start_even += default_duration
        start_odd = 1
        while start_odd + default_duration <= file_duration:
            command = "ffmpeg -ss " + str(start_odd) + " -i " + mp4_file_path + " -t " + str(
                default_duration)
            command += " -ab 160k -ac 1 -ar 44100 -vn " + wav_file_path + '_' + str(i) + '.wav'
            subprocess.call(command, shell=True)
            i += 1
            start_odd += default_duration

    @staticmethod
    def generate_files(mp4_file_path, wav_file_path, sentiment_folder_name, file_name):
        file_duration = WavGenerator.get_file_duration(mp4_file_path)
        wav_file_path = wav_file_path + sentiment_folder_name + file_name
        gap = constants.DEFAULT_DURATION - file_duration
        if gap == 0:
            command = "ffmpeg -i " + mp4_file_path
            command += " -ab 160k -ac 1 -ar 44100 -vn " + wav_file_path + '.wav'
            subprocess.call(command, shell=True)
            return
        if gap > 0:  # add padding
            command = "ffmpeg -i " + mp4_file_path
            command += " -af apad=pad_dur=" + str(gap)
            command += " -ab 160k -ac 1 -ar 44100 -vn " + wav_file_path + '.wav'
            subprocess.call(command, shell=True)
        else:
            # file duration is more than x seconds => need to split it
            WavGenerator.split_file(mp4_file_path, file_duration, wav_file_path)

    @staticmethod
    def generate_files_from_csv_and_dir(csv_file_name, mp4_files_directory, wav_files_directory):
        current_directory = os.path.dirname(__file__)
        df = pandas.read_csv(csv_file_name, usecols=[3, 5, 6])
        audio_data_dict = {}  # example : key="dia0_utt0" value="anger"

        for row in df.itertuples(index=True, name='Pandas'):
            file_name = "dia" + str(getattr(row, "Dialogue_ID")) + "_utt" + str(getattr(row, "Utterance_ID"))
            audio_data_dict[file_name] = getattr(row, "Emotion")

        wav_file_path = current_directory + '/' + wav_files_directory
        for relative_path in glob.glob(mp4_files_directory + '*.mp4'):
            file_name = relative_path.split('\\')[1].split('.')[0]
            if file_name in audio_data_dict:
                mp4_file_path = current_directory + '/' + mp4_files_directory + file_name + '.mp4'
                value = audio_data_dict[file_name]
                sentiment_folder_name = ""
                if value == "anger":
                    sentiment_folder_name = constants.FIRST_CLASS_DIR
                else:
                    sentiment_folder_name = constants.SECOND_CLASS_DIR
                WavGenerator.generate_files(mp4_file_path, wav_file_path, sentiment_folder_name, file_name)

    @staticmethod
    def generate_wav_files():
        wav_directory = constants.WAV_DIRECTORY
        print("GENERATING WAV FILES FOR TRAIN..")
        WavGenerator.generate_files_from_csv_and_dir(constants.TRAIN_CSV_FILE_NAME, constants.TRAIN_MP4_FILES_DIRECTORY, wav_directory + 'train/')

        print("GENERATING WAV FILES FOR TEST..")
        WavGenerator.generate_files_from_csv_and_dir(constants.TEST_CSV_FILE_NAME, constants.TEST_MP4_FILES_DIRECTORY, wav_directory + 'test/')

        print("GENERATING WAV FILES FOR VALIDATION..")
        WavGenerator.generate_files_from_csv_and_dir(constants.VALIDATION_CSV_FILE_NAME, constants.VALIDATION_MP4_FILES_DIRECTORY, wav_directory + 'validation/')




