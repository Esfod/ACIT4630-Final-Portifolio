import cv2
import os

def extract_bursts_from_video(input_dir, output_dir, burst_size=10, frame_stride=5, spacing=30):
    """
    Extracts a single burst from each video in input_dir.
    Each burst is saved in its own folder named after the video.

    Args:
        input_dir (str): Folder containing video files.
        output_dir (str): Folder where burst folders will be saved.
        burst_size (int): Number of frames to extract per burst.
        frame_stride (int): Frames to skip between saved frames within a burst.
        spacing (int): (Unused now, only one burst per video).
    """
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    total_videos = len(video_files)

    for video_index, filename in enumerate(video_files, start=1):
        video_path = os.path.join(input_dir, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open {video_path}")
            continue

        base_name = os.path.splitext(filename)[0]
        burst_folder = os.path.join(output_dir, f"{base_name}_burst")
        os.makedirs(burst_folder, exist_ok=True)

        print(f"📽 Processing video {video_index}/{total_videos}: {filename}")

        frames_saved = 0
        while frames_saved < burst_size:
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Incomplete burst at {burst_folder}, removing.")
                break
            frame_path = os.path.join(burst_folder, f"frame_{frames_saved:02d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames_saved += 1

            # Skip frames within burst
            for _ in range(frame_stride - 1):
                cap.grab()

        if frames_saved < burst_size:
            for f in os.listdir(burst_folder):
                os.remove(os.path.join(burst_folder, f))
            os.rmdir(burst_folder)
            print("🗑️ Incomplete burst deleted.")
        else:
            print(f"✅ Burst saved: {frames_saved} frames in {burst_folder}")

        cap.release()


# --- Configurable Parameters ---
burst_size = 10
frame_stride = 8
spacing = 30  # not used since only 1 burst is taken per video





# ----  Making custom Burst Dataset ----

if __name__ == "__main__":
    input_dir = "input_dir"             # folder with validation videos
    output_dir = "burst_validation_high_variation"          # where bursts should be saved

    extract_bursts_from_video(
        input_dir=input_dir,
        output_dir=output_dir,
        burst_size=10,
        frame_stride=5,
        spacing=30
    )

    print(f"✅ Extracted validation bursts from '{input_dir}' to '{output_dir}'")
