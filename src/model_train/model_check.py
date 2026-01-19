import torch

# 모델 경로 (사용자님 파일명으로 변경)
MODEL_PATH = "/Users/sungminhong/Documents/deepleaning_proj/checkpoints/best_model.pth"

try:
    # 1. 파일 로드
    print(f"📂 '{MODEL_PATH}' 로딩 중...")
    data = torch.load(MODEL_PATH, map_location='cpu')

    # 2. 데이터 타입 확인
    print(f"\n✅ 데이터 타입: {type(data)}")

    # 3. OrderedDict(가중치 딕셔너리)인 경우 내부 키 확인
    if isinstance(data, dict):
        print("ℹ️ 이 파일은 '모델 전체'가 아니라 '가중치(state_dict)'만 저장된 파일입니다.")
        print("-" * 30)
        keys = list(data.keys())
        print(f"총 레이어(Key) 개수: {len(keys)}개")
        print("\n[상위 10개 레이어 이름]")
        for k in keys[:10]:
            print(f" - {k}")
        print("\n..." )
        print("\n[마지막 5개 레이어 이름 (출력층 힌트)]")
        for k in keys[-5:]:
            print(f" - {k}")
            
    # 4. 모델 객체 자체인 경우
    else:
        print("ℹ️ 이 파일은 모델 객체 자체가 저장되어 있습니다.")
        print(data)

except Exception as e:
    print(f"❌ 읽기 실패: {e}")