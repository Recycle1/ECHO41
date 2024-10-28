import cv2
import numpy as np

def get_available_frame_indices(video_path):
    cap = cv2.VideoCapture(rf"{video_path}")
    available_indices = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        available_indices.append(frame_id)
        frame_id += 1

    cap.release()
    return available_indices

def process_video_with_opencv(video_path, clip_len=16):
    print(video_path)
    cap = cv2.VideoCapture(rf"{video_path}")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = len(get_available_frame_indices(video_path)) - 1
    # duration = cap.get(cv2.CAP_PROP_FPS) * frame_count
    interval = frame_count / (clip_len - 1)
    # indices = sample_frame_indices(clip_len, frame_sample_rate, frame_count)
    frames = []

    for i in range(clip_len):
        # Set the number of frames to capture
        frame_id = int(interval * i)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            print(f"Failed to retrieve frame at index {frame_id}")

    cap.release()

    if frames:
        stacked_frames = np.stack(frames)
        return stacked_frames
    else:
        print("No frames to stack.")
        return None