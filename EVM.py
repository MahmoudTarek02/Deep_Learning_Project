import os
import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack

# Function to load and split video into clips
def load_and_split_video(video_filename, clip_length=5):
    print(f"Loading video: {video_filename}")
    
    cap = cv2.VideoCapture(video_filename)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_per_clip = fps * clip_length
    num_clips = frame_count // frames_per_clip  # Number of full-length clips
    remainder_frames = frame_count % frames_per_clip  # Remaining frames for last clip

    video_clips = []
    clip = []
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        clip.append(frame)

        if (i + 1) % frames_per_clip == 0 or (i == frame_count - 1):  # Clip complete
            video_clips.append(np.array(clip, dtype='float'))
            clip = []

    cap.release()
    print(f"Split into {len(video_clips)} clips")
    
    return video_clips, fps, (width, height)

# Function to build Laplacian pyramid
def build_laplacian_pyramid(src, levels=3):
    gaussian_pyramid = [src.copy()]
    for _ in range(levels):
        gaussian_pyramid.append(cv2.pyrDown(gaussian_pyramid[-1]))

    pyramid = []
    for i in range(levels, 0, -1):
        GE = cv2.pyrUp(gaussian_pyramid[i])
        L = cv2.subtract(gaussian_pyramid[i-1], GE)
        pyramid.append(L)
    
    return pyramid

# Function to process video into Laplacian pyramid
def laplacian_video(video_tensor, levels=3):
    tensor_list = []
    for i in range(video_tensor.shape[0]):
        frame = video_tensor[i]
        pyr = build_laplacian_pyramid(frame, levels=levels)
        
        if i == 0:
            for k in range(levels):
                tensor_list.append(np.zeros((video_tensor.shape[0], pyr[k].shape[0], pyr[k].shape[1], 3)))
        
        for n in range(levels):
            tensor_list[n][i] = pyr[n]

    return tensor_list

# Butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y

# Function to reconstruct video
def reconstruct_from_tensorlist(filter_tensor_list, levels=3):
    final = np.zeros(filter_tensor_list[-1].shape)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels - 1):
            up = cv2.pyrUp(up) + filter_tensor_list[n + 1][i]
        final[i] = up
    return final

# Function to save video
def save_video(video_tensor, output_filename, fps):
    print(f"Saving processed video: {output_filename}")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height), 1)
    
    for i in range(video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    
    writer.release()
    print(f"Video saved: {output_filename}")

# Function to apply motion magnification
def magnify_motion(video_tensor, fps, output_path, low=0.4, high=3, levels=3, amplification=20):
    print(f"Applying motion magnification...")
    lap_video_list = laplacian_video(video_tensor, levels=levels)
    
    filter_tensor_list = []
    for i in range(levels):
        filter_tensor = butter_bandpass_filter(lap_video_list[i], low, high, fps)
        filter_tensor *= amplification
        filter_tensor_list.append(filter_tensor)

    recon = reconstruct_from_tensorlist(filter_tensor_list)
    final = video_tensor + recon

    save_video(final, output_path, fps)
    print(f"Motion magnification completed.")

# Function to process all videos in Data/ and save in Processed_Data
def process_all_videos():
    input_dir = "Data"
    output_dir = "Processed_Data"

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    print(f"Scanning directory: {input_dir}")

    for root, _, files in os.walk(input_dir):
        print(f"Checking directory: {root}")

        avi_files = [file for file in files if file.endswith(".avi")]

        if not avi_files:
            print(f"Warning: No video files found in {root}\n")
            continue  # Skip empty folders

        for file in avi_files:
            input_video_path = os.path.join(root, file)

            print(f"Found file: {input_video_path}")

            # Create corresponding output directory before processing
            relative_path = os.path.relpath(root, input_dir)
            output_folder = os.path.join(output_dir, relative_path, os.path.splitext(file)[0])
            os.makedirs(output_folder, exist_ok=True)

            # Process each clip **immediately** to reduce memory usage
            cap = cv2.VideoCapture(input_video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frames_per_clip = fps * 5  # Clip length: 5 seconds
            total_clips = frame_count // frames_per_clip  # Number of full clips

            print(f"Total clips: {total_clips}")

            for clip_idx in range(total_clips + 1):  # Process extra last clip if needed
                clip = []
                for _ in range(frames_per_clip):
                    ret, frame = cap.read()
                    if not ret:
                        break  # Stop if we run out of frames
                    clip.append(frame)

                if len(clip) == 0:
                    continue  # Skip empty clips

                # Convert to numpy array **without using float64**
                clip_array = np.array(clip, dtype='uint8')

                output_video = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_clip{clip_idx+1}_motion.avi")
                print(f"Processing Clip {clip_idx+1}/{total_clips} -> {output_video}")

                magnify_motion(clip_array, fps, output_video)

            cap.release()
            print(f"Completed processing for: {input_video_path}\n")

    print("Processing completed for all videos.")

# Run the processing
if __name__ == "__main__":
    process_all_videos()





