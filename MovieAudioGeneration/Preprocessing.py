from FeatureExtractor import save_combined_features, save_only_video_features

save_combined_features("Data\\train\\Videos", "Data\\train\\Audios", "Data\\Processed\\Combined", audio_sampling_rate=12000)
save_only_video_features("Data\\train\\Videos", "Data\\Processed\\Video(only)")