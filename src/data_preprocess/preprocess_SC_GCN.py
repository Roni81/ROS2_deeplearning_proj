import numpy as np
import os
import glob
import pickle
from tqdm import tqdm

def make_dataset_from_npy(input_dir, output_file, clip_length=300):
    """
    .npy 파일들이 있는 디렉토리에서 데이터를 읽어 ST-GCN용 (N, C, T, V, M) 데이터셋을 생성합니다.
    
    Args:
        input_dir (str): 'Assault', 'Drunken' 등 클래스 폴더가 있는 루트 디렉토리
        output_file (str): 저장할 파일 경로 (.pkl)
        clip_length (int): 고정할 프레임 길이 (Zero Padding)
    """
    
    # 클래스 정의 (디렉토리 이름과 매핑)
    # 디렉토리 이름은 대문자 시작이지만, 내부적으로 소문자로 처리하거나 매핑
    # 클래스 ID: 0:Assault, 1:Swoon, 2:Drunken, 3:Vandalism, 4:Normal
    class_map = {
        'Assault': 0, 
        'assault': 0,
        'Swoon': 1, 
        'swoon': 1,
        'Drunken': 2, 
        'drunken': 2, # Drunk -> Drunken (폴더명 기준)
        'drunk': 2,
        'Vandalism': 3,
        'vandalism': 3,
        'Normal': 4,
        'normal': 4
    }
    
    data_list = []
    label_list = []
    
    # 디렉토리 순회
    if not os.path.exists(input_dir):
        print(f"[오류] 입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        return

    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"[정보] 처리할 클래스 폴더: {subdirs}")
    
    for class_name in subdirs:
        if class_name not in class_map:
            print(f"[경고] 알 수 없는 클래스 폴더 무시: {class_name}")
            continue
            
        class_id = class_map[class_name]
        class_path = os.path.join(input_dir, class_name)
        file_list = glob.glob(os.path.join(class_path, "*.npy"))
        
        print(f"[진행] '{class_name}' 클래스 처리 중... (파일 {len(file_list)}개)")
        
        for npy_path in tqdm(file_list):
            try:
                # 1. 데이터 로드 (Frames, 34)
                data = np.load(npy_path, allow_pickle=True)
                
                # 차원 확인
                if len(data.shape) != 2 or data.shape[1] != 34:
                    # print(f"[스킵] 잘못된 데이터 형태: {os.path.basename(npy_path)} {data.shape}")
                    continue
                    
                T_origin, D = data.shape
                
                # 2. Reshape (Frames, 17, 2)
                # 정규화된 데이터는 xy 좌표만 있으므로 (17, 2)
                xy_data = data.reshape(T_origin, 17, 2)
                
                # 3. Confidence 채널 추가 -> (Frames, 17, 3)
                # 정규화된 데이터는 값이 존재하면 신뢰도 1, 없으면(0,0) 0으로 가정할 수 있으나
                # 여기서는 단순히 1로 채우거나, (0,0)인 경우 0으로 처리
                # xy_data가 (0,0)인 지점 찾기
                conf_data = np.ones((T_origin, 17, 1))
                # 좌표가 (0,0)인 경우 신뢰도 0으로 설정 (선택 사항)
                # mask = np.all(np.isclose(xy_data, 0), axis=2)
                # conf_data[mask] = 0
                
                xyz_data = np.concatenate((xy_data, conf_data), axis=2) # (T, 17, 3)
                
                # 4. Zero Padding / Cutting -> (clip_length, 17, 3)
                # ST-GCN은 (C, T, V, M) 순서
                
                # 먼저 (T, 17, 3) -> (3, T, 17) 로 Transpose (C, T, V)
                # C=3 (x, y, conf)
                # T=T_origin
                # V=17
                transposed = xyz_data.transpose(2, 0, 1) # (3, T, 17)
                
                # 최종 컨테이너 (3, clip_length, 17)
                sample_final = np.zeros((3, clip_length, 17))
                
                # 복사할 길이
                valid_len = min(T_origin, clip_length)
                
                # 데이터 채우기
                sample_final[:, :valid_len, :] = transposed[:, :valid_len, :]
                
                # 5. Person 차원 추가 (N, C, T, V, M)에서 M=1
                # sample_final: (C, T, V) -> (C, T, V, 1)
                sample_final = np.expand_dims(sample_final, axis=-1)
                
                data_list.append(sample_final)
                label_list.append(class_id)
                
            except Exception as e:
                print(f"[오류] 파일 처리 실패 {os.path.basename(npy_path)}: {e}")
                continue

    # Numpy 변환
    X_data = np.array(data_list)
    Y_data = np.array(label_list)
    
    print("\n[완료] 데이터셋 생성 완료")
    print(f"X Shape (데이터): {X_data.shape}  (N, C, T, V, M)")
    print(f"Y Shape (레이블): {Y_data.shape}  (N,)")
    
    # 저장
    save_dict = {'x_data': X_data, 'y_data': Y_data}
    with open(output_file, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"[저장] 데이터셋이 저장되었습니다: {output_file}")

if __name__ == '__main__':
    INPUT_DIR = "/Users/sungminhong/Documents/deepleaning_proj/datasets/train_data_1000_normalized"
    OUTPUT_FILE = "/Users/sungminhong/Documents/deepleaning_proj/datasets/stgcn_dataset.pkl"
    
    make_dataset_from_npy(INPUT_DIR, OUTPUT_FILE, clip_length=300)