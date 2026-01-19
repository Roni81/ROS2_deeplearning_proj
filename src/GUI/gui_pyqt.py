import sys
import cv2
import numpy as np
import os
import torch
from collections import deque
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from ultralytics import YOLO

# ★ 1단계에서 만든 model.py에서 클래스 임포트
try:
    from model import STGCN 
except ImportError:
    print("❌ 오류: 'model.py' 파일이 같은 폴더에 있어야 합니다.")
    sys.exit(1)

# ================= [설정] =================
VIDEO_PATH = "/Users/sungminhong/Documents/deepleaning_proj/datasets/vandal/87-6_cam02_vandalism01_place02_night_summer.mp4"       # 테스트할 영상 파일
YOLO_PATH = "/Users/sungminhong/Documents/deepleaning_proj/src/GUI/yolo11s-pose.pt"       # 욜로 모델
STGCN_WEIGHTS = "/Users/sungminhong/Documents/deepleaning_proj/checkpoints/best_model.pth" # 학습된 ST-GCN 가중치 경로

# ★ 중요: 학습시킬 때 데이터셋의 클래스 개수와 라벨을 정확히 맞춰주세요.
# 예: 0: Normal, 1: Swoon, 2: Assault ... (학습 데이터셋 순서 확인 필요)
ACTION_CLASSES = {
    0: "Assault",
    1: "Drunken",
    2: "Normal",
    3: "Swoon",
    4: "Vandalism"
}
NUM_CLASSES = len(ACTION_CLASSES)

# 추론에 사용할 윈도우 크기 (몇 프레임을 모아서 판단할지)
# 학습은 300으로 했더라도, 실시간성은 30~60 정도가 적당합니다.
# STGCN 모델의 AvgPool 덕분에 입력 길이가 달라도 작동합니다.
WINDOW_SIZE = 45 
# ==========================================

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_label_signal = pyqtSignal(str)

    def run(self):
        # 1. 디바이스 및 모델 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # YOLO 로드 (작은 사람도 잘 잡기 위해 이미지 사이즈 키움)
        yolo_model = YOLO(YOLO_PATH) if os.path.exists(YOLO_PATH) else YOLO("yolo11n-pose.pt")

        # ST-GCN 로드
        stgcn_model = None
        if os.path.exists(STGCN_WEIGHTS):
            try:
                stgcn_model = STGCN(num_classes=NUM_CLASSES)
                stgcn_model.to(device)
                stgcn_model.load_state_dict(torch.load(STGCN_WEIGHTS, map_location=device))
                stgcn_model.eval()
            except Exception:
                pass

        cap = cv2.VideoCapture(VIDEO_PATH)
        input_buffer = deque(maxlen=WINDOW_SIZE) # 예: 45

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                input_buffer.clear()
                continue

            # 기본 상태 메시지 (에러 방지용)
            current_status = "Scanning..." 

            # (A) YOLO 추론 (인식률 높이기 옵션 적용)
            results = yolo_model(frame, verbose=False, imgsz=1280, conf=0.2)
            annotated_frame = results[0].plot()

            # (B) 사람이 감지되었을 때
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                # ---------------------------------------------------------
                # 1. 데이터 추출 (이 부분이 있어야 버퍼에 넣을 수 있습니다)
                # ---------------------------------------------------------
                kpts = results[0].keypoints.xyn.cpu().numpy()[0] # (17, 2)
                conf = results[0].keypoints.conf.cpu().numpy()[0] # (17,)
                
                if conf is None: conf = np.ones((17,))
                
                # (3, 17) 형태로 만들기: [x, y, conf]
                person_data = np.vstack([kpts.T, conf[None, :]])
                
                # 버퍼에 추가
                input_buffer.append(person_data)

                # ---------------------------------------------------------
                # 2. 상태 결정 로직 (작성해주신 부분 적용)
                # ---------------------------------------------------------
                
                # [상황 1] 버퍼가 꽉 찼을 때 -> 추론 실행!
                if len(input_buffer) == WINDOW_SIZE:
                    if stgcn_model is not None:
                        # (Time, Channel, Node) -> (Channel, Time, Node)
                        data_np = np.array(input_buffer)
                        data_np = np.transpose(data_np, (1, 0, 2))
                        data_np = data_np[np.newaxis, :, :, :] # 배치 차원 추가
                        
                        inputs = torch.tensor(data_np, dtype=torch.float32).to(device)
                        
                        with torch.no_grad():
                            outputs = stgcn_model(inputs)
                            probs = torch.softmax(outputs, dim=1)
                            pred_idx = torch.argmax(probs, dim=1).item()
                            confidence = probs[0][pred_idx].item()
                            
                            label = ACTION_CLASSES.get(pred_idx, "Unknown")
                            current_status = f"{label} ({confidence*100:.0f}%)"
                    else:
                        current_status = "Model Error"

                # [상황 2] 아직 데이터 모으는 중일 때
                else:
                    current_status = f"Gathering... ({len(input_buffer)}/{WINDOW_SIZE})"

            else:
                # 사람이 안 보이면 "No Human" (버퍼는 유지하거나 비우거나 선택)
                current_status = "No Human"
                # input_buffer.clear() # 사람이 끊기면 처음부터 다시 모으려면 주석 해제

            # 화면 업데이트
            self.change_pixmap_signal.emit(annotated_frame)
            self.change_label_signal.emit(current_status)
            
            cv2.waitKey(1) # 속도 최적화

        cap.release()

class VideoWidgetApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ST-GCN Action Recognition System")
        self.resize(1280, 720)
        self.init_ui()

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_label_signal.connect(self.update_label)
        self.thread.start()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)

        self.overlay_label = QLabel("Initializing...", self)
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setFont(QFont("Arial", 40, QFont.Bold))
        self.overlay_label.resize(self.width(), 120)
        self.overlay_label.move(0, 40)

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.IgnoreAspectRatio
        )
        self.video_label.setPixmap(pixmap)

    def update_label(self, text):
        self.overlay_label.setText(text)
        
        # 텍스트 색상 처리
        color = "#FFFFFF"
        if "Normal" in text: color = "#00FF00"
        elif "Swoon" in text or "Assault" in text: color = "#FF0000"
        elif "Drunken" in text: color = "#FFA500"
        
        self.overlay_label.setStyleSheet(f"background-color: transparent; color: {color};")

    def resizeEvent(self, event):
        self.overlay_label.resize(self.width(), 120)
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWidgetApp()
    window.show()
    sys.exit(app.exec_())