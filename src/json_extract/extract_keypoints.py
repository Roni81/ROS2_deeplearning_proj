import cv2
import json
import re
import os
import glob
from ultralytics import YOLO
from tqdm import tqdm
import torch
import numpy as np
import time

# ================= 설정 =================
INPUT_DIR = "/Users/sungminhong/Documents/deepleaning_proj/datasets/drunken/add"
OUTPUT_DIR = "/Users/sungminhong/Documents/deepleaning_proj/datasets/drunken/add_json"
MODEL_NAME = 'yolo11s-pose.pt'

# [중요 변경] M3 Mac에서는 멀티프로세싱 + MPS 사용 시 프리징 현상이 잦으므로
# 단일 프로세스로 실행하는 것이 오히려 더 빠르고 안정적일 수 있습니다.
WORKER_COUNT = 2  

if torch.cuda.is_available():
    DEVICE = '0'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"시스템 감지: {DEVICE} 모드로 동작합니다.")

INFERENCE_ARGS = {
    'vid_stride': 1,       
    'imgsz': 640,          # [제안] 960/1280은 너무 큽니다. 속도를 위해 640 권장 (필요시 960 복구)
    'conf': 0.25,          
    'iou': 0.45,           
    'device': DEVICE,      
    'half': False,         
    'stream': True,
    'tracker': "bytetrack.yaml",
    'verbose': False       # YOLO 내부 로그 끄기
}
# =======================================

def process_single_video(video_path, output_path):
    try:
        # 이미 처리된 파일 스킵 기능 (옵션)
        if os.path.exists(output_path):
            return f"Skipped (Already exists): {os.path.basename(video_path)}"

        model = YOLO(MODEL_NAME)
        
        # 영상 정보 확인 (전체 프레임 수 계산용)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # 추적 시작
        results = model.track(source=video_path, persist=True, **INFERENCE_ARGS)
        
        frame_idx = 0
        stride = INFERENCE_ARGS['vid_stride']
        
        track_history = {} 
        temp_frames = [] 

        # [디버깅] 현재 파일 처리 시작 알림
        print(f"▶ Processing: {os.path.basename(video_path)} ({total_frames} frames)...")
        start_time = time.time()

        for result in results:
            # [진행 상황 체크] 100프레임마다 로그 출력 (멈춤 확인용)
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"   - {os.path.basename(video_path)}: {frame_idx}/{total_frames} ({elapsed:.1f}s)")

            real_frame_id = frame_idx * stride
            
            # 파일명 라벨 파싱 로직 (기존 동일)
            action_label = "unknown"
            try:
                filename = os.path.basename(video_path)
                parts = filename.split('_')
                if len(parts) > 2:
                    raw_action = parts[2]
                    match = re.search(r'([a-zA-Z]+)', raw_action)
                    if match:
                        action_label = match.group(1)
            except Exception:
                pass

            current_frame_info = { "frame_id": real_frame_id, "action": action_label, "people": [] }

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                keypoints_all = result.keypoints.data.cpu().numpy()
                
                for i, track_id in enumerate(track_ids):
                    x, y, w, h = boxes[i]
                    current_kpts = keypoints_all[i]
                    
                    if track_id not in track_history:
                        track_history[track_id] = {'positions': [], 'confs': []}
                    
                    track_history[track_id]['positions'].append((x, y))
                    kpt_conf_mean = np.mean(current_kpts[:, 2])
                    track_history[track_id]['confs'].append(kpt_conf_mean)

                    person_info = {
                        "person_index": i,
                        "track_id": track_id,
                        "box": [round(float(v), 2) for v in [x, y, w, h]],
                        "keypoints": [[round(float(val), 4) for val in k] for k in current_kpts]
                    }
                    current_frame_info["people"].append(person_info)

            temp_frames.append(current_frame_info)
            frame_idx += 1

        # --- [마네킹 필터링] ---
        noise_ids = set()
        for track_id, data in track_history.items():
            positions = np.array(data['positions'])
            confs = np.array(data['confs'])
            
            if len(positions) < 15:
                noise_ids.add(track_id)
                continue
            
            std_x = np.std(positions[:, 0])
            std_y = np.std(positions[:, 1])
            movement = std_x + std_y
            avg_conf = np.mean(confs)

            if movement < 10 and avg_conf < 0.6:
                noise_ids.add(track_id)
            if movement < 3:
                noise_ids.add(track_id)

        final_video_data = []
        for frame in temp_frames:
            filtered_people = []
            for p in frame["people"]:
                if p["track_id"] not in noise_ids:
                    filtered_people.append(p)
            frame["people"] = filtered_people
            final_video_data.append(frame)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(final_video_data, f, separators=(',', ':'))
            
        return f"Done: {os.path.basename(video_path)}"

    except Exception as e:
        return f"Error processing {os.path.basename(video_path)}: {str(e)}"

def main():
    video_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.mp4"), recursive=True)
    print(f"총 {len(video_files)}개의 영상을 처리합니다.")
    
    tasks = []
    for video_path in video_files:
        relative_path = os.path.relpath(video_path, INPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(relative_path)[0] + ".json")
        tasks.append((video_path, output_path))

    # [수정] 멀티프로세싱 제거하고 단순 반복문으로 변경 (MPS 안정성 확보)
    # 만약 꼭 멀티프로세싱을 쓰고 싶다면 WORKER_COUNT=1 로 하거나 DEVICE='cpu'로 변경해야 함
    
    print("\n--- [Single Process Mode] 시작 ---")
    for task in tqdm(tasks, desc="Total Progress"):
        result = process_single_video(task[0], task[1])
        # print(result) # 너무 시끄러우면 주석 처리

    print("\n🚀 처리 완료!")

if __name__ == "__main__":
    main()