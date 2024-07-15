import cv2
import os

def extract_frames(video_path, interval, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame interval based on the fps and the given interval in seconds
    frame_interval = int(fps * interval)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Save the frame as a JPG file
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames and saved to {output_folder}")

if __name__ == "__main__":
    video_path ="E:\\Random Python Scripts\\Debris-Segmentation\\One-Drive\\Aerial survey requests\\Aerial Survey Requests\\20230130\ASR1-Matanzas Pass Preserve\\Videos\DJI_20230130231334_0002_Z.MP4"
    interval = 5  # Seconds
    output_folder = "E:\\Random Python Scripts\\Debris-Segmentation\\Dataset\\Video Data\\Video 12"

    extract_frames(video_path, interval, output_folder)
