import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Get video file path from user input
input_video_path = input("Enter the path of the input video file (e.g., video.mp4): ")

# Load the template image for searching
template_image_path = "searching.png"
template_image = cv2.imread(template_image_path, cv2.IMREAD_COLOR)

# Convert the template image to grayscale for template matching
template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get the dimensions of the template image
template_height, template_width, _ = template_image.shape

# Initialize variables to track template match sections
sections = []
section_start_frame = None
section_end_frame = None

# Perform template matching for each frame with a loading bar
with tqdm(total=total_frames, desc="Processing Frames") as pbar:
    for frame_index in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for template matching
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply template matching
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8  # Adjust the threshold as needed
        locations = np.where(result >= threshold)

        # Check if template is found in the frame
        if len(locations[0]) > 0:
            # If section_start_frame is not set, set it to the current frame index
            if section_start_frame is None:
                section_start_frame = frame_index

            # Update the section_end_frame to the current frame index
            section_end_frame = frame_index

        else:
            # If section_start_frame is set, it means a template match section has ended
            if section_start_frame is not None:
                sections.append((section_start_frame, section_end_frame))
                section_start_frame = None
                section_end_frame = None

        pbar.update(1)

cap.release()

# Create video clips for non-template match sections
video_clips = []
start_frame = 0
for start_frame, end_frame in sections:
    if start_frame != 0:
        # Create a video clip for the non-template match section
        start_time = start_frame / fps
        end_time = end_frame / fps
        video_clip = VideoFileClip(input_video_path).subclip(start_time, end_time)
        video_clips.append(video_clip)

# Create a video clip for the remaining part of the video after the last template match
start_time = start_frame / fps
video_clip = VideoFileClip(input_video_path).subclip(start_time)
video_clips.append(video_clip)

# Concatenate the video clips into a single output video
output_video = concatenate_videoclips(video_clips)

# Generate the output video file
output_video_path = "output_video.mp4"
output_video.write_videofile(output_video_path, codec="libx264")

# Close the video clips
output_video.close()
