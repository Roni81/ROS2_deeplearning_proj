import cv2
import json
import os
import glob
from ultralytics import YOLO
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import multiprocessing
import numpy as np

# ================= 설정 =================
INPUT_DIR = "/Users/sungminhong/Documents/deepleaning_proj/datasets/swoon/temp"
OUTPUT_DIR = "/Users/sungminhong/Documents/deepleaning_proj/datasets/swoon/json_1_imgsz960_conf0.3_mamequin_filter"
MODEL_NAME = 'yolov8s-pose.pt'

if torch.cuda.is_available():
    DEVICE = '0'
elif torch.backends.mps.is_available():
    DEVICE = 'mps' # Mac M3는 여기 해당
else:
    DEVICE = 'cpu'

WORKER_COUNT = 2 # M3 발열 고려하여 2~3 추천

INFERENCE_ARGS = {
    'vid_stride': 1,       
    'imgsz': 960,         # M3 성능 믿고 1280 유지 (느리면 640으로 변경)
    'conf': 0.30,          # [수정] 0.46 -> 0.25 (겹친 사람도 잡기 위해 낮춤)
    'iou': 0.45,           # NMS 임계값
    'device': DEVICE,      
    'half': False,         # MPS에서는 False가 안정적일 수 있음
    'stream': True,
    'tracker': "bytetrack.yaml" # [추가] 가림 현상에 좀 더 강한 트래커
}
# =======================================

def process_single_video(file_info):
    try:
        video_path, output_path = file_info
        model = YOLO(MODEL_NAME)
        
        # 추적 시작
        results = model.track(source=video_path, persist=True, **INFERENCE_ARGS)
        
        frame_idx = 0
        stride = INFERENCE_ARGS['vid_stride']
        
        # 마네킹 필터링을 위한 기록 저장소
        # { track_id: {'positions': [], 'confs': []} }
        track_history = {} 
        temp_frames = [] 

        for result in results:
            real_frame_id = frame_idx * stride
            current_frame_info = { "frame_id": real_frame_id, "people": [] }

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                keypoints_all = result.keypoints.data.cpu().numpy() # (N, 17, 3)
                
                for i, track_id in enumerate(track_ids):
                    x, y, w, h = boxes[i]
                    current_kpts = keypoints_all[i] # [[x,y,conf], ...]
                    
                    # --- [데이터 축적: 마네킹 판별용] ---
                    if track_id not in track_history:
                        track_history[track_id] = {'positions': [], 'confs': []}
                    
                    # 중심 좌표 저장
                    track_history[track_id]['positions'].append((x, y))
                    
                    # 키포인트들의 평균 신뢰도(Confidence) 저장
                    # (conf 값이 있는 3번째 요소들의 평균)
                    kpt_conf_mean = np.mean(current_kpts[:, 2])
                    track_history[track_id]['confs'].append(kpt_conf_mean)

                    # --- [임시 저장] ---
                    person_info = {
                        "person_index": i,
                        "track_id": track_id,
                        "box": [round(float(v), 2) for v in [x, y, w, h]],
                        "keypoints": [[round(float(val), 4) for val in k] for k in current_kpts]
                    }
                    current_frame_info["people"].append(person_info)

            temp_frames.append(current_frame_info)
            frame_idx += 1

        # --- [강력해진 마네킹 필터링 로직] ---
        noise_ids = set()
        for track_id, data in track_history.items():
            positions = np.array(data['positions'])
            confs = np.array(data['confs'])
            
            # 조건 1: 등장 횟수가 너무 짧으면 노이즈 (예: 15프레임 미만)
            if len(positions) < 15:
                noise_ids.add(track_id)
                continue
            
            # 조건 2: 움직임이 거의 없으면 마네킹 (표준편차 활용)
            std_x = np.std(positions[:, 0])
            std_y = np.std(positions[:, 1])
            movement = std_x + std_y
            
            # 조건 3: 키포인트 신뢰도가 전반적으로 너무 낮으면 헛것(Ghost)
            avg_conf = np.mean(confs)

            # [판단] 움직임이 10 미만(매우 적음)이고, 평균 신뢰도도 0.6 미만이면 마네킹/배경 오인식으로 간주
            # (수치는 실제 데이터 보면서 조절 필요)
            if movement < 10 and avg_conf < 0.6:
                noise_ids.add(track_id)
            
            # [추가] 움직임이 아예 0에 수렴하면(완벽한 고정체) 신뢰도 상관없이 삭제
            if movement < 3:
                noise_ids.add(track_id)

        # 필터링 적용
        final_video_data = []
        for frame in temp_frames:
            filtered_people = []
            for p in frame["people"]:
                if p["track_id"] not in noise_ids:
                    filtered_people.append(p)
            
            # 사람이 있는 프레임만 저장할지, 전체 다 저장할지 선택 (여기선 전체 저장)
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

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with ProcessPoolExecutor(max_workers=WORKER_COUNT) as executor:
        futures = {executor.submit(process_single_video, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(video_files), desc="Processing"):
            result = future.result()
            if "Error" in result:
                print(f"\n🚨 {result}")

    print("\n🚀 처리 완료!")

if __name__ == "__main__":
    main()