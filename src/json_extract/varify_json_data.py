import os
import glob
import json

# 설정 (기존 경로 그대로)
INPUT_DIR = "/Users/sungminhong/Documents/deepleaning_proj/datasets/swoon/"
OUTPUT_DIR = "/Users/sungminhong/Documents/deepleaning_proj/datasets/swoon/entire_json"

def check_integrity():
    # 1. 파일 목록 가져오기
    video_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.mp4"), recursive=True)
    json_files = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.json"), recursive=True)
    
    # 2. 파일명(확장자 제외) 세트 만들기
    video_names = {os.path.splitext(os.path.basename(f))[0] for f in video_files}
    json_names = {os.path.splitext(os.path.basename(f))[0] for f in json_files}
    
    print(f"🎥 원본 영상 개수: {len(video_names)}")
    print(f"📄 생성된 JSON 개수: {len(json_names)}")
    
    # 3. 누락된 파일 찾기
    missing = video_names - json_names
    if missing:
        print(f"\n❌ [누락됨] 변환 안 된 영상 {len(missing)}개:")
        for m in list(missing)[:5]: # 5개만 예시로 출력
            print(f" - {m}.mp4")
    else:
        print("\n✅ 개수 일치! 누락된 파일 없습니다.")

    # 4. JSON 파일 내용 검사 (빈 파일이나 깨진 파일 확인)
    print("\n🔍 JSON 파일 무결성 검사 중...")
    corrupted = []
    for j_path in json_files:
        try:
            with open(j_path, 'r') as f:
                data = json.load(f)
                if not data: # 내용이 비어있음
                    corrupted.append(j_path)
        except json.JSONDecodeError:
            corrupted.append(j_path)
            
    if corrupted:
        print(f"❌ [손상됨] 내용이 비거나 깨진 파일 {len(corrupted)}개:")
        for c in corrupted:
            print(f" - {os.path.basename(c)}")
    else:
        print("✅ 모든 JSON 파일이 정상적으로 열립니다.")

if __name__ == "__main__":
    check_integrity()
    