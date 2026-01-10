import cv2
import json
import argparse
import numpy as np
import os
from tqdm import tqdm

# COCO Keypoint Index
# 0: Nose
# 1: Left Eye
# 2: Right Eye
# 3: Left Ear
# 4: Right Ear
# 5: Left Shoulder
# 6: Right Shoulder
# 7: Left Elbow
# 8: Right Elbow
# 9: Left Wrist
# 10: Right Wrist
# 11: Left Hip
# 12: Right Hip
# 13: Left Knee
# 14: Right Knee
# 15: Left Ankle
# 16: Right Ankle

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 11), (6, 12), (11, 12), # Torso
    (5, 7), (7, 9), (6, 8), (8, 10), # Arms
    (11, 13), (13, 15), (12, 14), (14, 16) # Legs
]

# Colors (BGR)
COLOR_HEAD = (0, 255, 255)    # Yellow
COLOR_TORSO = (255, 0, 0)     # Blue
COLOR_ARMS = (0, 255, 0)      # Green
COLOR_LEGS = (0, 0, 255)      # Red
COLOR_KEYPOINT = (0, 0, 0)    # Black dot

def get_connection_color(idx):
    if idx < 4: return COLOR_HEAD
    elif idx < 8: return COLOR_TORSO
    elif idx < 12: return COLOR_ARMS
    else: return COLOR_LEGS

def visualize(json_path, video_path=None, output_path=None):
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Organize data by frame_id
    frames_data = {item['frame_id']: item['people'] for item in data}
    max_frame_id = max(frames_data.keys()) if frames_data else 0
    
    # Video setup
    cap = None
    width, height = 1920, 1080 # Default
    fps = 30
    frame_count = max_frame_id + 150 # Add buffer

    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"▶ Loading video: {video_path} ({width}x{height} @ {fps}fps)")
    else:
        print(f"▶ Video not found or not provided. Using blank canvas ({width}x{height})")

    # Output setup
    if output_path is None:
        output_path = os.path.splitext(json_path)[0] + "_vis.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    current_people = [] # To hold last known state for strided frames (optional, but good for visualization smoothness)
    
    # Note: The JSON might have gaps due to 'vid_stride'. 
    # We can either just draw on exact frames, or hold the drawing.
    # extract_keypoints.py uses stride=3 by default. 
    # We will draw the keypoints on the exact frame ID, and for frames in between, 
    # we can optionally visualize the last known frame or just skip drawing (let's stick to exact frames for accuracy).

    for frame_idx in tqdm(range(frame_count), desc="Rendering"):
        frame = None
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255 # White background
        
        # Check if we have data for this frame
        if frame_idx in frames_data:
            people = frames_data[frame_idx]
            
            for person in people:
                kpts = person['keypoints']
                
                # Draw connections
                for i, (start_idx, end_idx) in enumerate(SKELETON_CONNECTIONS):
                    if start_idx >= len(kpts) or end_idx >= len(kpts): continue
                    
                    x1, y1, conf1 = kpts[start_idx]
                    x2, y2, conf2 = kpts[end_idx]
                    
                    if conf1 > 0.5 and conf2 > 0.5: # Visibility threshold
                        pt1 = (int(x1), int(y1))
                        pt2 = (int(x2), int(y2))
                        cv2.line(frame, pt1, pt2, get_connection_color(i), 2)
                
                # Draw keypoints
                for kp in kpts:
                    x, y, conf = kp
                    if conf > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 4, COLOR_KEYPOINT, -1)
        
        out.write(frame)

    if cap:
        cap.release()
    out.release()
    print(f"✅ Saved visualization to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pose keypoints from JSON")
    parser.add_argument("json_path", help="Path to the JSON file containing keypoints")
    parser.add_argument("--video", help="Path to the original video file (optional)", default=None)
    parser.add_argument("--out", help="Path to save the output video (optional)", default=None)
    
    args = parser.parse_args()
    
    # Auto-infer video path if not provided
    # Assumption: json is in dataset/swoon/json/NAME.json
    # video might be in dataset/swoon/temp/NAME.mp4
    video_path = args.video
    if video_path is None:
        # Try to guess
        base_name = os.path.splitext(os.path.basename(args.json_path))[0]
        # Common structure inferred from extract_keypoints.py
        # INPUT_DIR = ".../datasets/swoon/temp"
        # OUTPUT_DIR = ".../datasets/swoon/json"
        
        # Try to go up one level from 'json' and into 'temp'
        json_dir = os.path.dirname(os.path.abspath(args.json_path))
        parent_dir = os.path.dirname(json_dir)
        temp_dir = os.path.join(parent_dir, "temp")
        
        possible_video = os.path.join(temp_dir, base_name + ".mp4")
        if os.path.exists(possible_video):
            video_path = possible_video
            
    visualize(args.json_path, video_path, args.out)
