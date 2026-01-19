import json
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np

# Keypoint Names (COCO format)
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

def plot_keypoints(json_path, output_path=None):
    # 1. JSON 데이터 로드
    if not os.path.exists(json_path):
        print(f"Error: File not found - {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 2. 데이터 전처리
    frames = []
    # 17개 키포인트 각각에 대해 x, y, conf 리스트 초기화
    keypoints_over_time = {i: {'x': [], 'y': [], 'conf': []} for i in range(17)}
    
    # 프레임 순서대로 정렬
    sorted_data = sorted(data, key=lambda x: x['frame_id'])
    
    # 가장 많이 등장한 메인 Track ID 찾기 (데이터 끊김 방지용)
    # 기존 코드의 'person_index == 0'은 사람이 교차될 때 데이터가 섞일 수 있어 ID 기반으로 업그레이드했습니다.
    track_id_counts = {}
    for item in sorted_data:
        for p in item['people']:
            tid = p.get('track_id', p.get('person_index')) # track_id가 없으면 index 사용
            track_id_counts[tid] = track_id_counts.get(tid, 0) + 1
            
    if not track_id_counts:
        print("No people detected in the data.")
        return

    # 가장 자주 등장한 ID를 추적 대상으로 선정
    target_id = max(track_id_counts, key=track_id_counts.get)
    print(f"Tracking Target ID: {target_id} (Most frequent)")

    for item in sorted_data:
        frame_id = item['frame_id']
        frames.append(frame_id)
        
        # 현재 프레임에서 타겟 ID 찾기
        target_person = None
        for p in item['people']:
            # JSON에 track_id가 있으면 그것을, 없으면 person_index를 비교
            current_id = p.get('track_id', p.get('person_index'))
            if current_id == target_id:
                target_person = p
                break
        
        # 데이터 채우기 (발견 못했으면 NaN 처리)
        if target_person:
            kpts = target_person['keypoints']
            for i in range(17):
                if i < len(kpts):
                    # kpts[i]가 [x, y, conf] 혹은 [x, y] 일 수 있음
                    val = kpts[i]
                    x, y = val[0], val[1]
                    conf = val[2] if len(val) > 2 else 0.0
                    
                    keypoints_over_time[i]['x'].append(x)
                    keypoints_over_time[i]['y'].append(y)
                    keypoints_over_time[i]['conf'].append(conf)
                else:
                    keypoints_over_time[i]['x'].append(np.nan)
                    keypoints_over_time[i]['y'].append(np.nan)
                    keypoints_over_time[i]['conf'].append(0.0)
        else:
            for i in range(17):
                keypoints_over_time[i]['x'].append(np.nan)
                keypoints_over_time[i]['y'].append(np.nan)
                keypoints_over_time[i]['conf'].append(0.0)

    # 3. 그래프 그리기 (3행 2열)
    output_path = output_path or os.path.splitext(json_path)[0] + "_plot_xy.png"
    
    # figsize를 키워서(20, 18) 좌우 비교가 쉽도록 함
    fig, axs = plt.subplots(3, 2, figsize=(20, 18), sharex=True)
    fig.suptitle(f'Keypoint Trajectories (ID: {target_id}) - {os.path.basename(json_path)}', fontsize=16)
    
    # 그룹 정의
    groups = [
        ("Torso", [5, 6, 11, 12]), # Shoulders, Hips
        ("Arms", [7, 8, 9, 10]),   # Elbows, Wrists
        ("Legs", [13, 14, 15, 16]) # Knees, Ankles
    ]
    
    # 반복문으로 그래프 생성
    for row_idx, (group_name, kpt_indices) in enumerate(groups):
        # --- 왼쪽 컬럼: Y축 (Vertical) ---
        ax_y = axs[row_idx, 0]
        for k_idx in kpt_indices:
            ax_y.plot(frames, keypoints_over_time[k_idx]['y'], label=KEYPOINT_NAMES[k_idx], alpha=0.7, linewidth=1.5)
        
        ax_y.set_ylabel('Y Position (pixel)')
        ax_y.set_title(f'{group_name} - Vertical Movement (Y)')
        ax_y.legend(loc='upper right', fontsize='small')
        ax_y.grid(True, alpha=0.3)
        ax_y.invert_yaxis() # 이미지 좌표계: Y값이 커질수록 아래로 내려감 -> 그래프 반전

        # --- 오른쪽 컬럼: X축 (Horizontal) ---
        ax_x = axs[row_idx, 1]
        for k_idx in kpt_indices:
            ax_x.plot(frames, keypoints_over_time[k_idx]['x'], label=KEYPOINT_NAMES[k_idx], alpha=0.7, linewidth=1.5)
        
        ax_x.set_ylabel('X Position (pixel)')
        ax_x.set_title(f'{group_name} - Horizontal Movement (X)')
        ax_x.legend(loc='upper right', fontsize='small')
        ax_x.grid(True, alpha=0.3)
        # X축은 반전하지 않음

    plt.tight_layout()  # 그래프 간격 자동 조절 (제목 겹침 방지)
    plt.savefig(output_path) # 실제 파일로 저장하는 핵심 코드
    plt.close() # 메모리 해제
    
    print(f"Plot saved successfully: {output_path}")


if __name__ == "__main__":
    # json_path를 본인의 실제 json 파일 경로로 바꿔주세요
    json_file = "/Users/sungminhong/Documents/deepleaning_proj/datasets/vandal/json/88-2_cam02_vandalism01_place09_day_spring.json"
    plot_keypoints(json_file)