import torch
from transformers import ASTModel, VideoMAEModel, VideoMAEImageProcessor, ASTFeatureExtractor
import librosa
import torch.nn.functional as F
from pathlib import Path
from VideoProcessing import process_video_with_opencv
def interpolate_features(input_tensor, size):
    input_tensor_permuted = input_tensor.permute(0, 2, 1)
    resized_tensor_permuted = F.interpolate(input_tensor_permuted, size=size, mode='linear', align_corners=False)
    resized_tensor = resized_tensor_permuted.permute(0, 2, 1)
    return resized_tensor

def save_combined_features(input_video_dir, input_audio_dir, output_dir, audio_sampling_rate = 12000):
    input_video_path = Path(input_video_dir)
    input_audio_path = Path(input_audio_dir)
    output_path = Path(output_dir)

    files = list(input_video_path.rglob('*'))
    total_files = len(files)

    video_processor = VideoMAEImageProcessor.from_pretrained("PretrainedModel/videomae-base-finetuned-kinetics/preprocessor_config.json")
    video_model = VideoMAEModel.from_pretrained("./PretrainedModel/videomae-base-finetuned-kinetics")
    audio_processor = ASTFeatureExtractor.from_pretrained("PretrainedModel/ast-finetuned-audioset-10-10-0.4593/preprocessor_config.json", sampling_rate=audio_sampling_rate)
    audio_model = ASTModel.from_pretrained("./PretrainedModel/ast-finetuned-audioset-10-10-0.4593")

    output_path.mkdir(parents=True, exist_ok=True)

    i = 1

    # Go through each subdirectory
    for category_video_dir, category_audio_dir in zip(input_video_path.iterdir(), input_audio_path.iterdir()):
        if category_video_dir.is_dir():
            # Create an output subdirectory for each category
            category_output_dir = output_path / category_video_dir.name
            category_output_dir.mkdir(exist_ok=True)

            # Go through all the files in the subdirectory
            for video, audio in zip(category_video_dir.glob('*'), category_audio_dir.glob('*')):
                print(f"({i} / {total_files}): {video.name} | {audio.name}")

                # Save the processed file to a new directory
                output_file_path = category_output_dir / video.stem

                audio_array, _ = librosa.load(audio, sr=audio_sampling_rate)
                # audio file is decoded on the fly
                audio_inputs = audio_processor(audio_array, sampling_rate=audio_sampling_rate, return_tensors="pt")
                with torch.no_grad():
                    audio_outputs = audio_model(**audio_inputs)
                last_hidden_state_audio = audio_outputs.last_hidden_state

                video_sample = process_video_with_opencv(video, clip_len=16)
                # prepare video for the model
                video_inputs = video_processor(list(video_sample), return_tensors="pt")
                with torch.no_grad():
                    video_outputs = video_model(**video_inputs)
                last_hidden_state_video = video_outputs.last_hidden_state

                # The temporal dimension is aligned using temporal interpolation
                aligned_audio_resampled = interpolate_features(last_hidden_state_audio, last_hidden_state_video.size(1))

                combined_features = torch.cat((last_hidden_state_video, aligned_audio_resampled), dim=2)
                combined_features = torch.squeeze(combined_features)
                torch.save(combined_features, f"{output_file_path}.pt")

                i += 1

def save_only_video_features(input_video_dir, output_dir):
    input_video_path = Path(input_video_dir)
    output_path = Path(output_dir)

    files = list(input_video_path.rglob('*'))
    total_files = len(files)

    video_processor = VideoMAEImageProcessor.from_pretrained("PretrainedModel/videomae-base-finetuned-kinetics/preprocessor_config.json")
    video_model = VideoMAEModel.from_pretrained("./PretrainedModel/videomae-base-finetuned-kinetics")

    # Create an output directory if it does not exist
    output_path.mkdir(parents=True, exist_ok=True)

    i = 1

    # Go through each subdirectory
    for category_video_dir in input_video_path.iterdir():
        if category_video_dir.is_dir():
            # Create an output subdirectory for each category
            category_output_dir = output_path / category_video_dir.name
            category_output_dir.mkdir(exist_ok=True)

            # Iterate through all the files in the subdirectory
            for video in category_video_dir.glob('*'):
                print(f"({i} / {total_files}): {video.name}")

                # Save the processed file to a new directory
                output_file_path = category_output_dir / video.stem
                video_sample = process_video_with_opencv(video, clip_len=16)
                # prepare video for the model
                video_inputs = video_processor(list(video_sample), return_tensors="pt")

                with torch.no_grad():
                    video_outputs = video_model(**video_inputs)

                last_hidden_state_video = video_outputs.last_hidden_state
                last_hidden_state_video = torch.squeeze(last_hidden_state_video)
                torch.save(last_hidden_state_video, f"{output_file_path}.pt")

                i += 1
