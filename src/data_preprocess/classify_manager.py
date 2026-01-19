import json
import numpy as np
import matplotlib.pyplot as plt
import os

def find_best_clip_by_class(json_path, target_class='vandalism', clip_length=300):
    """
    [기능]
    JSON 파일의 데이터를 분석하여 지정한 행동(target_class) 점수가 
    가장 높은 구간(clip_length 프레임)을 찾고 그래프로 보여줍니다.
    """
    
    # 1. 파일 존재 여부 확인
    if not os.path.exists(json_path):
        print(f"Error: 파일을 찾을 수 없습니다 -> {json_path}")
        print("파일 경로를 다시 확인해주세요.")
        return 0, 0

    # 2. JSON 로드
    print(f"데이터 로딩 중... ({json_path})")
    with open(json_path, 'r', encoding='utf-8') as f: # 한글 경로 등 대비 utf-8 추가
        data = json.load(f)
    
    # 프레임 순서 정렬 (혹시 섞여 있을 경우 대비)
    data = sorted(data, key=lambda x: x['frame_id'])
    
    total_frames = len(data)
    print(f"총 프레임 수: {total_frames}")

    # 데이터가 클립 길이보다 짧으면 전체 반환
    if total_frames <= clip_length:
        print(f"Warning: 데이터 길이({total_frames})가 클립 길이({clip_length})보다 짧습니다.")
        return 0, total_frames

    # 3. 타겟 ID 선정 (가장 많이 등장한 사람 추적)
    track_counts = {}
    for item in data:
        for p in item['people']:
            # track_id 우선, 없으면 person_index 사용
            tid = p.get('track_id', p.get('person_index'))
            track_counts[tid] = track_counts.get(tid, 0) + 1
            
    if not track_counts: 
        print("Error: 사람 데이터가 없습니다.")
        return 0, 0
    
    target_id = max(track_counts, key=track_counts.get)
    print(f"분석 대상 ID: {target_id} (가장 많이 등장함)")
    print(f"분석 클래스: [{target_class}]")

    # 4. 프레임별 점수 계산 로직
    scores = []
    frames = []
    prev_kpts = None
    
    for item in data:
        frame_id = item['frame_id']
        frames.append(frame_id)
        
        # 타겟 사람 찾기
        person = next((p for p in item['people'] if p.get('track_id', p.get('person_index')) == target_id), None)
        
        score = 0.0
        
        if person and prev_kpts is not None:
            # (x, y) 좌표만 추출
            curr_kpts = np.array([k[:2] for k in person['keypoints']])
            
            # 관절 이동 거리 (속도)
            diff = np.linalg.norm(curr_kpts - prev_kpts, axis=1) 
            
            # --- [기물파손 (Vandalism) 로직] ---
            if target_class == 'vandalism':
                wrist_speed = np.sum(diff[[9, 10]])  # 손목
                ankle_speed = np.sum(diff[[15, 16]]) # 발목
                torso_speed = np.mean(diff[[5, 6, 11, 12]]) # 몸통
                
                # 몸통은 안정적인데 손발이 폭발적인 경우 감지
                impact_energy = (wrist_speed * 1.5) + (ankle_speed * 2.0)
                score = impact_energy - (torso_speed * 0.5)
                if score < 0: score = 0

            # --- [실신 (Swoon) 로직] ---
            elif target_class == 'swoon':
                head_drop = curr_kpts[0][1] - prev_kpts[0][1]
                avg_drop_velocity = head_drop # 단순화
                if avg_drop_velocity > 2.0:
                    score = avg_drop_velocity * 5.0
                else: score = 0

            # --- [주취 (Drunk) 로직] ---
            elif target_class == 'drunk':
                hip_center_x = np.mean(curr_kpts[[11, 12], 0])
                nose_x = curr_kpts[0][0]
                sway = abs(nose_x - hip_center_x)
                score = sway

            # --- [폭행 (Assault) 로직] ---
            elif target_class == 'assault':
                arm_mvmt = np.sum(diff[[7, 8, 9, 10]]) 
                score = arm_mvmt

        if person:
            prev_kpts = np.array([k[:2] for k in person['keypoints']])
            
        scores.append(score)

    # 5. 최적 구간 찾기 (Moving Average)
    if len(scores) < clip_length:
         return 0, len(data)

    window_sum = np.convolve(scores, np.ones(clip_length), 'valid')
    best_start_idx = np.argmax(window_sum)
    best_end_idx = best_start_idx + clip_length
    
    start_f = frames[best_start_idx]
    # 인덱스 범위 초과 방지
    end_idx_safe = min(best_end_idx - 1, len(frames) - 1)
    end_f = frames[end_idx_safe]
    
    print(f"--------------------------------------------------")
    print(f"[{target_class}] 분석 결과")
    print(f"최고 점수 구간: Frame {start_f} ~ {end_f}")
    print(f"--------------------------------------------------")
    
    # 6. 시각화
    plt.figure(figsize=(12, 5))
    x_axis = frames[clip_length-1:]
    
    if len(x_axis) == len(window_sum):
        plt.plot(x_axis, window_sum, label=f'{target_class} Score', color='red')
        plt.axvspan(start_f, end_f, color='yellow', alpha=0.3, label='Best Clip')
        plt.title(f'Action Detection: {target_class.upper()}')
        plt.xlabel('Frame ID')
        plt.ylabel('Score (Sum)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return start_f, end_f

# =========================================================
# [사용 방법] 아래 경로만 수정해서 실행하세요
# =========================================================
if __name__ == "__main__":
    
    # 1. 여기에 가지고 계신 JSON 파일의 경로를 적어주세요.
    # (같은 폴더에 있다면 파일명만, 다른 폴더면 전체 경로 입력)
    my_json_file = "/Users/sungminhong/Documents/deepleaning_proj/datasets/vandal/json/87-1_cam01_vandalism01_place02_night_summer.json" # <--- 여기 수정!

    # 2. 분석할 행동 설정 ('vandalism', 'swoon', 'drunk', 'assault')
    target_action = 'vandalism' 

    # 3. 추출할 길이 (프레임 수, 30fps 기준 300 = 10초)
    clip_len = 300 

    # 실행
    start_frame, end_frame = find_best_clip_by_class(my_json_file, target_action, clip_len)