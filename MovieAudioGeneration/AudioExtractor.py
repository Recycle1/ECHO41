from pathlib import Path
from transformers import AutoProcessor, ASTModel
import torch
import librosa
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def process_audio_features(input_audio_dir, output_dir, audio_sampling_rate = 16000):
    input_audio_path = Path(input_audio_dir)
    output_path = Path(output_dir)

    files = list(input_audio_path.rglob('*'))
    total_files = len(files)

    audio_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    audio_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    output_path.mkdir(parents=True, exist_ok=True)

    i = 1

    # Go through each subdirectory
    for category_audio_dir in input_audio_path.iterdir():
        if category_audio_dir.is_dir():
            # Create an output subdirectory for each category
            category_output_dir = output_path / category_audio_dir.name
            category_output_dir.mkdir(exist_ok=True)

            # Iterate through all the files in the subdirectory
            for audio in category_audio_dir.glob('*'):
                print(f"({i} / {total_files}): {audio.name}")

                # Save the processed file to a new directory
                output_file_path = category_output_dir / audio.stem

                audio, sr = librosa.load(audio, sr=audio_sampling_rate)

                audio_inputs = audio_processor(audio, sampling_rate=sr, return_tensors="pt")
                with torch.no_grad():
                    outputs = audio_model(**audio_inputs)

                audio_features = outputs.last_hidden_state
                audio_features = torch.squeeze(audio_features)
                torch.save(audio_features, f"{output_file_path}.pt")

                i += 1

process_audio_features("Data\\train\\Audios", "Data\\Processed\\Audio(only)")