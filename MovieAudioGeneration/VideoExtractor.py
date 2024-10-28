import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor
from pathlib import Path
from VideoProcessing import process_video_with_opencv
import numpy as np

def process_video_features(input_video_dir, output_dir):
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

def process_video_features2(input_video_dir, output_file_path):
    input_video_path = Path(input_video_dir)
    video_features = []

    files = list(input_video_path.rglob('*'))
    total_files = len(files)

    video_processor = VideoMAEImageProcessor.from_pretrained("PretrainedModel/videomae-base-finetuned-kinetics/preprocessor_config.json")
    video_model = VideoMAEModel.from_pretrained("./PretrainedModel/videomae-base-finetuned-kinetics")

    i = 1

    # Go through each subdirectory
    for category_video_dir in input_video_path.iterdir():
        if category_video_dir.is_dir():

            # Iterate through all the files in the subdirectory
            for video in category_video_dir.glob('*'):
                print(f"({i} / {total_files}): {video.name}")
                video_sample = process_video_with_opencv(video, clip_len=16)
                # prepare video for the model
                video_inputs = video_processor(list(video_sample), return_tensors="pt")

                with torch.no_grad():
                    video_outputs = video_model(**video_inputs)

                last_hidden_state_video = video_outputs.last_hidden_state
                last_hidden_state_video = torch.squeeze(last_hidden_state_video)
                video_features.append(last_hidden_state_video)

                i += 1

    video_features_matrix = np.stack(video_features)
    np.save(output_file_path, video_features_matrix)



# process_video_features("Data\\train\\Videos", "Data\\Processed\\Video(only)")
process_video_features2("Data\\train\\Videos", "./Data/video_features.npy")


