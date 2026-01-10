import numpy as np

def normalize_keypoints(data):
    """
    JSON 데이터를 받아 Body-Scale Normalization을 수행합니다.
    1. Translation Invariance: 골반 중심을 (0,0)으로 이동
    2. Scale Invariance: 몸통 길이(Neck~MidHip)를 기준으로 크기 정규화
    """
    normalized_data = []

    # COCO Keypoint Index
    # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LSho, 6:RSho, 
    # 7:LElb, 8:RElb, 9:LWri, 10:RWri, 11:LHip, 12:RHip, ...
    
    for frame in data:
        new_frame = frame.copy()
        new_people = []
        
        for person in frame['people']:
            kpts = np.array(person['keypoints']).reshape(-1, 3) # [x, y, conf]
            
            # 신뢰도(Confidence)만 따로 뺌
            xy = kpts[:, :2] 
            confs = kpts[:, 2:3]

            # 1. 기준점(Root) 설정: 양쪽 골반(11, 12)의 중간 지점
            # 만약 골반이 안 보이면 어깨(5, 6) 중간 사용
            if (xy[11][0] > 0 and xy[12][0] > 0):
                root = (xy[11] + xy[12]) / 2.0
            elif (xy[5][0] > 0 and xy[6][0] > 0):
                root = (xy[5] + xy[6]) / 2.0
            else:
                # 기준점이 없으면 정규화 불가 -> 원본 유지 혹은 건너뜀
                new_people.append(person)
                continue

            # 2. 중심 이동 (Root-Centering)
            xy_centered = xy - root

            # 3. 스케일 기준(Scale Reference) 계산: 몸통 길이 (Neck ~ Root)
            # Neck은 어깨(5,6)의 중간점으로 추정
            if (xy[5][0] > 0 and xy[6][0] > 0):
                neck = (xy[5] + xy[6]) / 2.0
                # 원래 좌표계에서 길이 계산
                torso_len = np.linalg.norm(neck - root)
            else:
                torso_len = 0

            # 몸통 길이가 유효하지 않으면(너무 작거나 0), 
            # 머리(0)부터 골반까지 거리 혹은 임의의 값 사용
            if torso_len < 10: 
                # 대안: 전체 관절 중 root에서 가장 먼 거리 사용 (Max Spread)
                torso_len = np.max(np.linalg.norm(xy_centered, axis=1))
                if torso_len == 0: torso_len = 1 # 0 나누기 방지

            # 4. 스케일 정규화 (Scaling)
            xy_normalized = xy_centered / torso_len

            # 5. 데이터 합치기 [x_norm, y_norm, conf]
            kpts_norm = np.hstack((xy_normalized, confs)).flatten().tolist()
            
            new_person = person.copy()
            new_person['keypoints'] = kpts_norm
            new_people.append(new_person)
        
        new_frame['people'] = new_people
        normalized_data.append(new_frame)
        
    return normalized_data