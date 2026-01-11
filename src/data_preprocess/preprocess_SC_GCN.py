import numpy as np
import json
import os

# 1. 앞서 만든 함수들 (그대로 사용)
# from your_module import find_best_clip_by_class, normalize_keypoints

def make_dataset_for_stgcn(file_list, target_class='assault', clip_length=300):
    """
    JSON 파일 리스트를 받아 ST-GCN용 (N, C, T, V, M) 데이터를 만듭니다.
    """
    data_list = []
    label_list = []
    
    # 클래스 ID 매핑 (예시)
    class_map = {'assault': 0, 'swoon': 1, 'drunk': 2, 'vandalism': 3}
    class_id = class_map[target_class]

    for json_path in file_list:
        # 1. 최적의 구간(Start/End Frame) 찾기
        start_f, end_f = find_best_clip_by_class(json_path, target_class, clip_length)
        
        # 데이터가 너무 짧거나 없으면 스킵
        if start_f == 0 and end_f == 0: continue
            
        # 2. JSON 다시 로드하여 해당 구간만 자르기
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        # 프레임 ID 기준 정렬
        raw_data.sort(key=lambda x: x['frame_id'])
        
        # 구간 슬라이싱 (프레임 ID가 아니라 리스트 인덱스로 접근)
        # 실제로는 frame_id와 인덱스가 다를 수 있으니 frame_id 매칭 로직 필요
        # 여기서는 간단히 리스트 슬라이싱으로 가정
        clip_data = [d for d in raw_data if start_f <= d['frame_id'] <= end_f]
        
        # 300프레임보다 부족하면 Zero Padding (뒤쪽 채우기)
        while len(clip_data) < clip_length:
            # 빈 프레임 추가 (구조만 유지)
            empty_frame = clip_data[-1].copy() if clip_data else raw_data[0].copy()
            empty_frame['people'] = [] # 사람 없음 처리
            clip_data.append(empty_frame)

        # 3. 정규화 (Normalization)
        # 이전에 만든 함수 사용: 중심 이동 및 스케일링 된 데이터 반환
        normalized_clip = normalize_keypoints(clip_data)
        
        # 4. (C, T, V, M) 형태로 변환
        # 초기화: (3채널, 300프레임, 17관절, 1사람)
        sample_npy = np.zeros((3, clip_length, 17, 1))
        
        for t, frame in enumerate(normalized_clip):
            if t >= clip_length: break
            
            # 타겟 사람 1명만 가져옴 (정규화 함수에서 이미 처리되었다고 가정)
            if frame['people']:
                person = frame['people'][0] # 첫 번째 사람 (Target)
                
                # normalize_keypoints 결과가 1차원 리스트([x,y,c, x,y,c...])라고 가정하면:
                kpts = np.array(person['keypoints']).reshape(17, 3) # (17, 3)
                
                # Transpose: (17, 3) -> (3, 17) 로 바꿔서 넣어야 함
                # sample_npy의 구조: [channel, time, vertex, person]
                sample_npy[0, t, :, 0] = kpts[:, 0] # X 좌표
                sample_npy[1, t, :, 0] = kpts[:, 1] # Y 좌표
                sample_npy[2, t, :, 0] = kpts[:, 2] # Confidence
        
        data_list.append(sample_npy)
        label_list.append(class_id)

    # 5. 최종 결과: Numpy Array 변환
    # (N, C, T, V, M) 형태
    X_data = np.array(data_list) 
    Y_data = np.array(label_list)
    
    print(f"Dataset Created for {target_class}")
    print(f"X Shape: {X_data.shape}") # 예: (100, 3, 300, 17, 1)
    print(f"Y Shape: {Y_data.shape}")
    
    return X_data, Y_data