import torch
import os
from torch.utils.data import Dataset
import librosa
import numpy as np

def standardize(mel_spectrogram):
    mean = np.mean(mel_spectrogram)
    std = np.std(mel_spectrogram)
    standardized_mel_spectrogram = (mel_spectrogram - mean) / std
    return standardized_mel_spectrogram

class FeatureDataset(Dataset):

    def __init__(self, features_dir, targets_dir, target_type, sampling_rate = 12000):

        self.features_samples = []
        self.targets_samples = []
        self.target_type = target_type
        self.sampling_rate = sampling_rate

        # Assume all .pt files have a corresponding .wav file with the same name
        for subdir, _, files in os.walk(features_dir):
            for file in files:
                if file.endswith('.pt'):
                    feature_path = os.path.join(subdir, file)
                    relative_path = os.path.relpath(feature_path, features_dir)
                    target_file_name = os.path.splitext(file)[0] + '.wav'
                    target_path = os.path.join(targets_dir, os.path.dirname(relative_path), target_file_name)

                    self.features_samples.append(feature_path)
                    self.targets_samples.append(target_path)

    def __len__(self):
        return len(self.features_samples)

    def __getitem__(self, idx):

        feature_path = self.features_samples[idx]
        target_path = self.targets_samples[idx]

        # Load feature and audio
        feature = torch.load(feature_path)
        audio_array, sr = librosa.load(target_path)

        if self.target_type == "wav":
            audio_resampled = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sampling_rate)
            expected_samples = int(len(audio_array) * self.sampling_rate / sr)

            if len(audio_resampled) < expected_samples:
                audio_resampled = np.pad(audio_resampled, (0, expected_samples - len(audio_resampled)), mode='constant')
            elif len(audio_resampled) > expected_samples:
                audio_resampled = audio_resampled[:expected_samples]

            audio_resampled = audio_resampled * 10000

            return feature, audio_resampled
        elif self.target_type == "mel":
            S1 = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=256, n_fft=2048, hop_length=512)
            S_DB1 = librosa.power_to_db(S1, ref=np.max)
            S_DB1 = standardize(S_DB1)

            return feature, S_DB1
