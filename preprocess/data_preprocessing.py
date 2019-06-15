
from preprocess.generate_wav import WavGenerator
from preprocess.generate_spectrograms import SpecGenerator


class DataPreprocessing:

    @staticmethod
    def preprocess_data(generate_wav, generate_spec):
        if generate_wav:
            WavGenerator.generate_wav_files()
        if generate_spec:
            SpecGenerator.generate_spectrograms()


DataPreprocessing.preprocess_data(True, True)