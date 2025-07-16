import os
from moviepy import VideoFileClip

def convert_avi_to_mp4(input_avi_path, output_mp4_path):
    """
    Converts an AVI video file to MP4 format.

    Args:
        input_avi_path (str): The path to the input AVI file.
        output_mp4_path (str): The path for the output MP4 file.
    """
    try:
        clip = VideoFileClip(input_avi_path)
        clip.write_videofile(output_mp4_path, codec="libx264", audio_codec="aac")
        print(f"Conversion successful: {input_avi_path} -> {output_mp4_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")


directory = "finals_verification/Testing_Jul_14"

for vid in os.scandir(directory):
    file_name = os.path.basename(vid.path)[:-4]
    convert_avi_to_mp4(vid.path, f'finals_verification/mp4_vids/{file_name}.mp4')  