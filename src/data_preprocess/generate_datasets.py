import os
import json
import numpy as np
import pandas as pd
import glob
import random
from tqdm import tqdm

# ================= [설정: 경로 및 파라미터] =================
INPUT_DIR = "/Users/sungminhong/Documents/deepleaning_proj/datasets/raw_json"
OUTPUT_DIR = "/Users/sungminhong/Documents/deepleaning_proj/datasets/train_data_150_1" # 폴더명 변경 권장

FPS = 30
WINDOW_SIZE = 150       
PRE_EVENT_RATIO = 0.8   
SCREEN_H = 960         
NORMAL_SAMPLE_RATIO = 1.0 

# ================= [강화된 물리 트리거 알고리즘] =================

def check_swoon(df, idx):
    """
    [실신 강화] 
    1. 낙하 속도가 매우 빨라야 함 (털썩)
    2. 단순히 낮아지는 게 아니라 '가로'로 길어져야 함 (Aspect Ratio)
    3. [핵심] 떨어지고 나서 일정 시간 동안 '바닥에 머물러야' 함 (앉았다 일어나는 것 제외)
    """
    # 1. 시점 설정 (0.5초 전과 비교 - 순간적인 낙하 감지)
    past_idx = idx - 15 
    future_idx = idx + 45 # 1.5초 후 (일어나지 않는지 확인용)
    
    if past_idx < 0 or future_idx >= len(df): return False
    
    # 2. Y좌표(높이) 변화 확인
    curr_head = df.iloc[idx]['y0']
    past_head = df.iloc[past_idx]['y0']
    
    # [조건 1] 낙하 속도 강화: 화면의 10% 이상 쿵 떨어져야 함 (기존 7% -> 10% 상향)
    drop_dist = curr_head - past_head
    if drop_dist < (SCREEN_H * 0.10): return False

    # 3. 자세 무너짐 (Bounding Box 비율) 확인
    row = df.iloc[idx]
    ys = row[1::2].values; ys = ys[ys > 0]
    xs = row[0::2].values; xs = xs[xs > 0]
    
    if len(ys) < 5 or len(xs) < 5: return False # 키포인트 너무 적으면 패스

    h_box = ys.max() - ys.min()
    w_box = xs.max() - xs.min()
    
    # [조건 2] 완전히 눕거나 찌그러짐 (너비가 높이보다 크거나, 높이가 극도로 낮음)
    # 앉아있는 건 보통 1:1 비율이므로, 1.3배 이상 가로가 길어야 함
    is_flat = (h_box > 0) and (w_box / h_box > 1.3)
    is_crumbled = h_box < (SCREEN_H * 0.2) # 높이가 화면의 20% 미만으로 찌그러짐

    if not (is_flat or is_crumbled): return False

    # [조건 3 - 핵심] Stay Down Check (넘어진 후 1.5초 뒤에도 여전히 낮아야 함)
    future_head = df.iloc[future_idx]['y0']
    # 머리 위치가 회복되지 않고 바닥권(화면 절반 아래)에 머물러야 함
    if future_head < (SCREEN_H * 0.5): # 다시 올라왔다는 뜻 (값이 작을수록 위)
        return False
        
    return True

def check_assault(df, idx):
    """
    [폭행 강화]
    단순 빠른 움직임(달리기) 제외.
    '몸통은 고정된 상태에서 손/발만 폭발적으로 나가는' 펀치/킥 동작 감지.
    """
    past_idx = idx - 5 # 아주 짧은 순간 (임팩트)
    if past_idx < 0: return False
    
    def get_velocity(indices):
        dists = []
        for i in indices:
            cx, cy = df.iloc[idx][f'x{i}'], df.iloc[idx][f'y{i}']
            px, py = df.iloc[past_idx][f'x{i}'], df.iloc[past_idx][f'y{i}']
            if cx==0 or px==0: continue
            dists.append(np.sqrt((cx-px)**2 + (cy-py)**2))
        return np.max(dists) if dists else 0

    # 손목(9,10), 발목(15,16)의 속도
    limb_vel = get_velocity([9, 10, 15, 16])
    # 몸통(5,6,11,12)의 속도
    body_vel = get_velocity([5, 6, 11, 12])

    # [조건 1] 타격 속도 대폭 상향 (화면의 8% 이상 이동/5프레임)
    if limb_vel < (SCREEN_H * 0.08): return False

    # [조건 2] 상대 속도 (Limb가 Body보다 훨씬 빨라야 함)
    # 달리기는 몸통도 같이 빨라짐. 펀치/킥은 몸통 대비 사지가 2.5배 이상 빨라야 함.
    if body_vel > 0 and (limb_vel / body_vel < 2.5): return False
    
    return True

def check_vandalism(df, idx):
    """
    [기물파손 강화]
    한 번의 움직임이 아니라 '반복적이고 격렬한' 진동이 있어야 함.
    """
    window = 15
    start = idx - window
    if start < 0: return False

    # 15프레임 동안의 손목/발목 이동 거리 누적
    cum_dist = 0
    for i in range(start, idx):
        # 양쪽 손목 중 더 많이 움직인 쪽 기준
        curr_r, curr_l = df.iloc[i][['x10','y10']], df.iloc[i][['x9','y9']]
        prev_r, prev_l = df.iloc[i-1][['x10','y10']], df.iloc[i-1][['x9','y9']]
        
        d_r = np.linalg.norm(curr_r - prev_r)
        d_l = np.linalg.norm(curr_l - prev_l)
        cum_dist += max(d_r, d_l)

    # [조건 1] 누적 움직임이 커야 함 (지속성)
    if cum_dist < (SCREEN_H * 0.25): return False

    # [조건 2] 수직/수평 범위가 넓어야 함 (단순 걷기 제외, 팔을 휘두르는 범위)
    chunk = df.iloc[start:idx+1]
    # 손목 좌표
    wrist_xs = pd.concat([chunk['x9'], chunk['x10']])
    wrist_ys = pd.concat([chunk['y9'], chunk['y10']])
    valid_w = wrist_xs[wrist_xs > 0]
    
    if len(valid_w) == 0: return False
    
    box_area = (wrist_xs.max() - wrist_xs.min()) * (wrist_ys.max() - wrist_ys.min())
    # 좁은 영역에서 꼼지락거리는 것 제외 (넓게 휘둘러야 함)
    if box_area < (SCREEN_H * SCREEN_H * 0.02): return False
    
    return True

def check_drunken(df, idx):
    """
    [주취 강화]
    걷는 것(Zigzag)보다 '제자리에서 중심을 못 잡는(Sway)' 상태에 집중.
    발은 좁은 범위에 있는데, 머리가 크게 흔들리면 주취일 확률 높음.
    """
    window = 60 # 2초
    start = idx - window
    if start < 0: return False
    
    chunk = df.iloc[start : idx+1]
    
    # 1. 발의 움직임 범위 (Bbox of Ankles)
    ankles_x = pd.concat([chunk['x15'], chunk['x16']])
    ankles_x = ankles_x[ankles_x > 0]
    if len(ankles_x) < window: return False # 발이 안 보이면 패스
    
    foot_spread = ankles_x.max() - ankles_x.min()
    
    # [조건 1] 발은 거의 제자리거나 좁게 움직임 (걸어가는 중이면 제외 - 걷는 건 정상일 수 있음)
    if foot_spread > (SCREEN_H * 0.2): return False 

    # 2. 머리의 움직임 범위 (COG Sway)
    head_x = chunk['x0'][chunk['x0'] > 0]
    if len(head_x) < window: return False
    
    head_mean = head_x.mean()
    head_deviation = np.abs(head_x - head_mean).mean() # 머리가 평균 위치에서 얼마나 벗어나는가

    # [조건 2] 머리의 흔들림이 심함
    if head_deviation > (SCREEN_H * 0.05): # 중심축에서 5% 이상 계속 왔다갔다 함
        return True
        
    return False

# ================= [메인 로직 (기존과 동일하되 함수 연결)] =================

def main():
    # [설정] 폴더명과 트리거 함수 매핑
    CLASS_MAP = {
        "Swoon": check_swoon,
        "Assault": check_assault,
        "Vandalism": check_vandalism,
        "Drunken": check_drunken
    }
    
    # 기존 코드와 동일한 루프 구조 유지
    # (OUTPUT_DIR 폴더 생성 및 stats 초기화 등)
    for cls in list(CLASS_MAP.keys()) + ["Normal"]:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)
    stats = {k: 0 for k in list(CLASS_MAP.keys()) + ["Normal"]}

    for label, trigger_func in CLASS_MAP.items():
        label_dir = os.path.join(INPUT_DIR, label)
        if not os.path.exists(label_dir):
            continue
            
        files = glob.glob(os.path.join(label_dir, "*.json"))
        print(f"▶ [{label}] 분석 시작 (Strict Mode)...")

        for fpath in tqdm(files, desc=f"Processing {label}"):
            try:
                with open(fpath, 'r') as f: data = json.load(f)
            except: continue
            if not data: continue
            
            # --- (트랙킹 ID 추출 및 보간 로직은 기존과 동일) ---
            t_counts = {}
            for fr in data:
                for p in fr['people']:
                    tid = p.get('track_id', -1)
                    if tid != -1: t_counts[tid] = t_counts.get(tid, 0) + 1
            if not t_counts: continue
            main_tid = max(t_counts, key=t_counts.get)
            
            raw = []
            for fr in data:
                row = [0]*34
                tgt = next((p for p in fr['people'] if p.get('track_id')==main_tid), None)
                if tgt:
                    flat = []
                    for k in tgt['keypoints']: flat.extend([k[0], k[1]])
                    row = flat
                raw.append(row)
            cols = [f"{a}{i}" for i in range(17) for a in ['x','y']]
            df = pd.DataFrame(raw, columns=cols)
            df = df.replace(0, np.nan).interpolate(limit_direction='both').fillna(0)
            if len(df) < WINDOW_SIZE: continue
            
            event_mask = np.zeros(len(df), dtype=int)
            events_found = 0
            
            # --- [수정된 부분: 윈도우 슬라이딩 간격 최적화] ---
            # 너무 촘촘하게 검사하면 중복 데이터가 많아지므로 5프레임 단위로 검사
            idx = WINDOW_SIZE
            while idx < len(df) - 60:
                if trigger_func(df, idx):
                    pre = int(WINDOW_SIZE * PRE_EVENT_RATIO)
                    start = idx - pre
                    end = start + WINDOW_SIZE
                    
                    if start >= 0 and end <= len(df):
                        npy = df.iloc[start:end].values
                        base_name = os.path.splitext(os.path.basename(fpath))[0]
                        fname = f"{base_name}_{idx}.npy"
                        
                        save_path = os.path.join(OUTPUT_DIR, label, fname)
                        np.save(save_path, npy)
                        
                        stats[label] += 1
                        events_found += 1
                        
                        safe_s = max(0, start - 90) # 마스킹 범위 확대
                        safe_e = min(len(df), end + 90)
                        event_mask[safe_s : safe_e] = 1
                        
                        # 이벤트를 하나 찾으면 과감하게 점프 (중복 방지)
                        idx += WINDOW_SIZE // 2 
                        continue
                
                idx += 5 # 검사 간격을 2 -> 5로 늘려 연산 속도 및 중복 방지
                
            # --- [Normal 추출 로직 동일] ---
            target_normal = max(1, int(events_found * NORMAL_SAMPLE_RATIO))
            clean_indices = []
            # Normal도 띄엄띄엄 추출
            for i in range(0, len(df) - WINDOW_SIZE, 60):
                if np.sum(event_mask[i : i + WINDOW_SIZE]) == 0:
                    clean_indices.append(i)
                    
            if clean_indices:
                selected = random.sample(clean_indices, min(len(clean_indices), target_normal))
                for s_idx in selected:
                    npy = df.iloc[s_idx : s_idx + WINDOW_SIZE].values
                    base_name = os.path.splitext(os.path.basename(fpath))[0]
                    fname = f"{base_name}_normal_{s_idx}.npy"
                    np.save(os.path.join(OUTPUT_DIR, "Normal", fname), npy)
                    stats["Normal"] += 1

    print("\n✅ 강화된 데이터셋 생성 완료!")
    for k, v in stats.items():
        print(f"{k}: {v}개")

if __name__ == "__main__":
    main()