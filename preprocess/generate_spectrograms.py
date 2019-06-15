import glob
import os
import librosa
import matplotlib.pyplot as plot
import numpy as np
from preprocess import constants


class SpecGenerator:

    @staticmethod
    def shift_signal(data, shift_value):
        return np.roll(data, shift_value)

    @staticmethod
    def augment_instance(signal_data, freq, spectrograms_path, img_name):
        count = 1
        for shifting_value in constants.SHIFTING_VALUES:
            data_shifted = SpecGenerator.shift_signal(signal_data, shifting_value)
            plot.title('Spectrogram')
            plot.specgram(data_shifted, Fs=freq, NFFT=512)
            plot.xlabel('Time')
            plot.ylabel('Frequency')
            plot.xlim([0, constants.DEFAULT_DURATION])
            suffix = '_tr_' + str(count)
            tr_image_path = spectrograms_path + img_name + suffix + ".png"
            plot.savefig(tr_image_path)
            plot.clf()
            count += 1

    @staticmethod
    def generate_spectrograms_for_dataset(wav_files_path, spectrograms_path, is_train_data):
        for file_path in glob.glob(wav_files_path + '*.wav'):
            signal_data, freq = librosa.load(file_path, sr=None)
            path_elements = file_path.split('\\')
            path_el_length = len(path_elements)
            img_name = path_elements[path_el_length - 1].split('.')[0]  # ex: dia5_utt4
            if is_train_data:  #augmentation
                SpecGenerator.augment_instance(signal_data, freq, spectrograms_path, img_name)
            plot.title('Spectrogram')
            plot.specgram(signal_data, Fs=freq, NFFT=512)
            plot.xlabel('Time')
            plot.ylabel('Frequency')
            plot.xlim([0, constants.DEFAULT_DURATION])
            image_path = spectrograms_path + img_name + ".png"
            plot.savefig(image_path)
            plot.clf()

    @staticmethod
    def generate_spectrograms():
        current_directory = os.path.dirname(__file__)
        wav_directory_path = current_directory + "/" + constants.WAV_DIRECTORY
        spectograms_path = current_directory.replace("preprocess", "") + "/" + constants.SPECTROGRAMS_DIRECTORY
        train_subdir = constants.TRAIN_SUBDIR
        test_subdir = constants.TEST_SUBDIR
        valid_subdir = constants.VALIDATION_SUBDIR
        first_class = constants.FIRST_CLASS_DIR
        second_class = constants.SECOND_CLASS_DIR

        print("GENERATING SPECTROGRAMS FOR TRAIN...")
        SpecGenerator.generate_spectrograms_for_dataset(wav_directory_path + train_subdir + first_class,
                            spectograms_path + train_subdir + first_class, True)
        SpecGenerator.generate_spectrograms_for_dataset(wav_directory_path + train_subdir + second_class,
                            spectograms_path + train_subdir + second_class, True)

        print("GENERATING SPECTROGRAMS FOR TEST...")
        SpecGenerator.generate_spectrograms_for_dataset(wav_directory_path + test_subdir + first_class,
                            spectograms_path + test_subdir + first_class, False)
        SpecGenerator.generate_spectrograms_for_dataset(wav_directory_path + test_subdir + second_class,
                            spectograms_path + test_subdir + second_class, False)

        print("GENERATING SPECTROGRAMS FOR VALIDATION...")
        SpecGenerator.generate_spectrograms_for_dataset(wav_directory_path + valid_subdir + first_class,
                            spectograms_path + valid_subdir + first_class, False)
        SpecGenerator.generate_spectrograms_for_dataset(wav_directory_path + valid_subdir + second_class,
                            spectograms_path + valid_subdir + second_class, False)

