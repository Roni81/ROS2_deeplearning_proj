import json
import numpy as np
import matplotlib.pyplot as plt

def find_best_clip_by_class(json_path, target_class='vandalism', clip_length=300):
    """
    지원 클래스: 'assault'(폭행), 'swoon'(실신), 'drunk'(주취), 'vandalism'(기물파손)
    """
    
    # 1. JSON 로드 및 정렬
    with open(json_path, 'r') as f:
        data = json.load(f)
    data = sorted(data, key=lambda x: x['frame_id'])
    
    if len(data) <= clip_length:
        return 0, len(data)

    # 2. 타겟 ID 선정
    track_counts = {}
    for item in data:
        for p in item['people']:
            tid = p.get('track_id', p.get('person_index'))
            track_counts[tid] = track_counts.get(tid, 0) + 1
            
    if not track_counts: return 0, 0
    target_id = max(track_counts, key=track_counts.get)
    
    print(f"Target ID: {target_id}, Analyzing for class: [{target_class}]")

    # 3. 프레임별 점수 계산
    scores = []
    frames = []
    prev_kpts = None
    
    for item in data:
        frame_id = item['frame_id']
        frames.append(frame_id)
        
        person = next((p for p in item['people'] if p.get('track_id', p.get('person_index')) == target_id), None)
        
        score = 0.0
        
        if person and prev_kpts is not None:
            curr_kpts = np.array([k[:2] for k in person['keypoints']])
            
            # 각 관절의 순간 이동 거리 (속도)
            diff = np.linalg.norm(curr_kpts - prev_kpts, axis=1) 
            
            # =========================================================
            # [4. 기물파손 (Vandalism)] - 타격 및 발차기 감지
            # =========================================================
            if target_class == 'vandalism':
                # (1) 사지의 폭발적 속도 (Explosive Velocity)
                # 걷거나 뛸 때보다 훨씬 빠른 속도로 손/발이 움직이는 구간
                wrist_speed = np.sum(diff[[9, 10]])  # 양쪽 손목
                ankle_speed = np.sum(diff[[15, 16]]) # 양쪽 발목
                
                # (2) 몸통 안정성 대비 사지 움직임 (Limb-to-Body Ratio)
                # 몸은 가만히 있는데 발만 뻗거나(발차기), 팔만 휘두르는(망치질) 경우
                torso_speed = np.mean(diff[[5, 6, 11, 12]]) # 어깨, 골반 평균 속도
                
                # 기물파손 특징: 몸통 속도에 비해 손발 속도가 압도적으로 빠름
                # 가중치: 발차기(Ankle)와 휘두르기(Wrist)에 집중
                impact_energy = (wrist_speed * 1.5) + (ankle_speed * 2.0)
                
                # 단순히 뛰는 것(Running)과 구별하기 위해 몸통 속도 차감 (선택사항)
                # 뛰면 몸통도 같이 빨라지지만, 기물파손은 타격 부위만 엄청 빠름
                score = impact_energy - (torso_speed * 0.5)
                
                if score < 0: score = 0 # 음수 방지

            # =========================================================
            # [기존 클래스 로직 유지]
            # =========================================================
            elif target_class == 'swoon':
                head_drop = curr_kpts[0][1] - prev_kpts[0][1]
                shoulder_drop = np.mean(curr_kpts[[5,6], 1]) - np.mean(prev_kpts[[5,6], 1])
                avg_drop_velocity = (head_drop + shoulder_drop) / 2.0
                
                prev_height = np.mean(prev_kpts[[15,16], 1]) - prev_kpts[0][1]
                curr_height = np.mean(curr_kpts[[15,16], 1]) - curr_kpts[0][1]
                height_loss = prev_height - curr_height
                
                if avg_drop_velocity > 2.0:
                    score = (avg_drop_velocity * 5.0) + (height_loss * 2.0)
                else: score = 0.0

            elif target_class == 'drunk':
                hip_center_x = np.mean(curr_kpts[[11, 12], 0])
                nose_x = curr_kpts[0][0]
                lean_offset = abs(nose_x - hip_center_x)
                sway_velocity = abs(curr_kpts[0][0] - prev_kpts[0][0])
                leg_speed = np.mean(diff[[15, 16]])
                
                if leg_speed > 0.5: 
                    score = (lean_offset * 1.5) + (sway_velocity * 3.0)
                else:
                    score = sway_velocity * 2.0

            elif target_class == 'assault':
                arm_mvmt = np.sum(diff[[7, 8, 9, 10]]) 
                body_mvmt = np.sum(diff[[5, 6, 11, 12]])
                score = (arm_mvmt * 2.0) + (body_mvmt * 1.0)

        if person:
            prev_kpts = np.array([k[:2] for k in person['keypoints']])
            
        scores.append(score)

    # 4. 결과 도출 (Moving Average)
    window_sum = np.convolve(scores, np.ones(clip_length), 'valid')
    best_start_idx = np.argmax(window_sum)
    best_end_idx = best_start_idx + clip_length
    
    start_f = frames[best_start_idx]
    end_f = frames[best_end_idx - 1] if best_end_idx < len(frames) else frames[-1]
    
    print(f"[{target_class}] 최적 구간: Frame {start_f} ~ {end_f}")
    
    # 시각화
    plt.figure(figsize=(10, 4))
    plt.plot(frames[clip_length-1:], window_sum, label=f'{target_class} Score', color='purple')
    plt.axvspan(start_f, end_f, color='violet', alpha=0.3, label='Selected Clip')
    plt.title(f'Event Detection: {target_class.upper()}')
    plt.legend()
    plt.show()

    return start_f, end_f