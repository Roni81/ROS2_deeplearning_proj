import numpy as np
import os
import glob
import random
from tqdm import tqdm

def normalize_keypoints(data):
    """
    Legacy function for JSON data.
    JSON 데이터를 받아 Body-Scale Normalization을 수행합니다.
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
                new_people.append(person)
                continue

            # 2. 중심 이동 (Root-Centering)
            xy_centered = xy - root

            # 3. 스케일 기준(Scale Reference) 계산: 몸통 길이 (Neck ~ Root)
            if (xy[5][0] > 0 and xy[6][0] > 0):
                neck = (xy[5] + xy[6]) / 2.0
                torso_len = np.linalg.norm(neck - root)
            else:
                torso_len = 0

            if torso_len < 10: 
                torso_len = np.max(np.linalg.norm(xy_centered, axis=1))
                if torso_len == 0: torso_len = 1 

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

def normalize_single_frame(kpts_xy):
    """
    Normalize a single frame of keypoints (17, 2).
    """
    # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LSho, 6:RSho, 
    # 7:LElb, 8:RElb, 9:LWri, 10:RWri, 11:LHip, 12:RHip, 
    # 13:LKnee, 14:RKnee, 15:LAnk, 16:RAnk
    
    # 1. Root Calculation (Mid-Hip)
    # Note: Assuming 0 is missing value, but normalization often results in 0. 
    # In .npy data, usually 0,0 means missing if raw. 
    # But checking > 0 condition is safer for raw pixel coords.
    
    # Check if hips are visible (not significantly close to 0,0)
    # Using a small epsilon or strictly > 0 if coordinates are positive pixels
    l_hip = kpts_xy[11]
    r_hip = kpts_xy[12]
    
    if np.any(l_hip > 0) and np.any(r_hip > 0):
        root = (l_hip + r_hip) / 2.0
    else:
        # Fallback to shoulders
        l_sho = kpts_xy[5]
        r_sho = kpts_xy[6]
        if np.any(l_sho > 0) and np.any(r_sho > 0):
            root = (l_sho + r_sho) / 2.0
        else:
            # If no root, cannot normalize properly. Return as is or zeros?
            # Existing logic returns as is roughly.
            # Let's try to find center of gravity of all visible points
            visible_points = kpts_xy[np.where((kpts_xy[:,0] > 0) & (kpts_xy[:,1] > 0))]
            if len(visible_points) > 0:
                root = np.mean(visible_points, axis=0)
            else:
                return kpts_xy # All zero or invalid
    
    # 2. Centering
    xy_centered = kpts_xy - root
    
    # 3. Scale Reference (Torso Length)
    l_sho = kpts_xy[5]
    r_sho = kpts_xy[6]
    
    torso_len = 0
    if np.any(l_sho > 0) and np.any(r_sho > 0):
        neck = (l_sho + r_sho) / 2.0
        torso_len = np.linalg.norm(neck - root)
        
    if torso_len < 1e-6: # Too small or zero
         # Max spread from root
         torso_len = np.max(np.linalg.norm(xy_centered, axis=1))
         if torso_len == 0: torso_len = 1.0
         
    # 4. Scaling
    xy_normalized = xy_centered / torso_len
    
    return xy_normalized

def normalize_numpy_data(data):
    """
    Processes numpy array of shape (F, 34).
    Returns normalized data of shape (F, 34).
    """
    F, D = data.shape
    if D != 34:
        raise ValueError(f"Expected dim 34, got {D}")
        
    reshaped_data = data.reshape(F, 17, 2)
    normalized_frames = []
    
    for i in range(F):
        frame_kpts = reshaped_data[i]
        norm_kpts = normalize_single_frame(frame_kpts)
        normalized_frames.append(norm_kpts)
        
    return np.array(normalized_frames).reshape(F, 34)

def process_datasets(input_root, output_root, target_count=1000):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    classes = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    for cls in classes:
        print(f"Processing class: {cls}")
        cls_input_dir = os.path.join(input_root, cls)
        cls_output_dir = os.path.join(output_root, cls)
        
        if not os.path.exists(cls_output_dir):
            os.makedirs(cls_output_dir)
            
        files = glob.glob(os.path.join(cls_input_dir, "*.npy"))
        file_count = len(files)
        print(f"  Found {file_count} files.")
        
        if file_count == 0:
            continue
            
        # Balancing with fixed seed for reproducibility
        random.seed(42) 
        
        selected_files = []
        if file_count < target_count:
            # Upsample (Replacement)
            print(f"  Upsampling from {file_count} to {target_count}...")
            # Ensure we include all original files at least once
            selected_files = files.copy()
            # method 1: random choice for remaining
            remaining_needed = target_count - file_count
            selected_files.extend(random.choices(files, k=remaining_needed))
        else:
            # Downsample (No replacement)
            print(f"  Downsampling from {file_count} to {target_count}...")
            selected_files = random.sample(files, target_count)
            
        print(f"  Normalizing and saving {len(selected_files)} files...")
        
        for i, fpath in enumerate(tqdm(selected_files)):
            basename = os.path.basename(fpath)
            # Handle duplicates from upsampling by appending index if needed
            # Or just overwrite? 
            # If we upsample, we have multiple copies of the same file content.
            # We should probably save them with unique names to avoid overwriting or valid dataset size.
            # E.g. filename_0.npy, filename_1.npy
            
            # Since 'selected_files' can contain duplicates (upsampling), 
            # and we are iterating them, we must ensure unique output filenames.
            
            # Strategy: if duplicate, append suffix
            name, ext = os.path.splitext(basename)
            save_name = f"{name}_{i}{ext}" # unique index for every file in the list
            
            try:
                data = np.load(fpath, allow_pickle=True)
                # Check shape
                if len(data.shape) == 2 and data.shape[1] == 34:
                     norm_data = normalize_numpy_data(data)
                     save_path = os.path.join(cls_output_dir, save_name)
                     np.save(save_path, norm_data)
                else:
                    print(f"    Skipping {basename}: Invalid shape {data.shape}")
            except Exception as e:
                print(f"    Error processing {basename}: {e}")

if __name__ == "__main__":
    INPUT_DIR = "/Users/sungminhong/Documents/deepleaning_proj/datasets/train_data_150"
    OUTPUT_DIR = "/Users/sungminhong/Documents/deepleaning_proj/datasets/train_data_1000_normalized"
    
    process_datasets(INPUT_DIR, OUTPUT_DIR, target_count=1000)
    print("Done!")