import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# ================= [설정] =================
# 학습된 모델 파일 경로 (어제 저장한 파일명 확인!)
MODEL_PATH = "./checkpoints/best_model.pth" 
# 클래스 순서 (학습할 때의 폴더 알파벳 순서와 정확히 일치해야 함)
# 예: Assault, Drunken, Normal, Swoon, Vandalism
CLASSES = ["Assault", "Drunken", "Normal", "Swoon", "Vandalism"]

# 데이터 설정 (학습 때와 동일하게)
WINDOW_SIZE = 300  # 5초 (60fps 기준)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# ================= [1. 모델 클래스 정의] =================
# (학습 코드의 모델 클래스를 그대로 복사해와야 가중치를 로드할 수 있습니다)
class Graph:
    def __init__(self):
        self.edges = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                      (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        self.A = self.get_A()
    def get_A(self):
        A = np.zeros((17, 17))
        for i, j in self.edges: A[i,j]=1; A[j,i]=1
        A = A + np.eye(17)
        D = np.sum(A, axis=0)
        D_inv = np.diag(D**(-0.5))
        D_inv[np.isinf(D_inv)] = 0
        adj = np.dot(np.dot(D_inv, A), D_inv)
        return torch.tensor(adj, dtype=torch.float32).to(DEVICE)

class GCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1)
        self.A = A
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, (9,1), (stride,1), padding=(4,0)),
            nn.BatchNorm2d(out_c), nn.Dropout(0.5)
        )
        self.res = nn.Sequential(nn.Conv2d(in_c, out_c, 1, (stride,1)), nn.BatchNorm2d(out_c)) if in_c!=out_c or stride!=1 else lambda x:x
    def forward(self, x):
        res = self.res(x)
        x = self.conv(x)
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        return F.relu(self.tcn(x) + res)

class STGCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.graph = Graph()
        self.bn = nn.BatchNorm1d(34)
        self.layers = nn.ModuleList([
            GCNBlock(2, 64, self.graph.A),
            GCNBlock(64, 64, self.graph.A),
            GCNBlock(64, 128, self.graph.A, 2),
            GCNBlock(128, 256, self.graph.A, 2)
        ])
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0,3,1,2).contiguous().view(N, V*C, T)
        x = self.bn(x)
        x = x.view(N,V,C,T).permute(0,2,3,1).contiguous()
        for l in self.layers: x = l(x)
        x = F.avg_pool2d(x, x.size()[2:])
        return self.fc(x.view(N, -1))

# ================= [2. AI 처리 스레드] =================
class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_label_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        # 모델 로드
        print("▶ 모델 로딩 중...")
        self.stgcn = STGCN(num_classes=len(CLASSES)).to(DEVICE)
        self.stgcn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.stgcn.eval()
        
        self.yolo = YOLO('yolov8n-pose.pt') # 또는 yolov8m-pose.pt
        print("▶ 모델 로딩 완료!")

        # 데이터 버퍼 (Sliding Window)
        self.frame_buffer = deque(maxlen=WINDOW_SIZE)

    def normalize_keypoints(self, kpts):
        """ 학습 때와 똑같은 정규화 로직 적용 """
        # kpts shape: (17, 2)
        l_hip = kpts[11]
        r_hip = kpts[12]
        center = (l_hip + r_hip) / 2
        return kpts - center # 원점 이동

    def run(self):
        # 웹캠 0번 (파일 테스트 시: cap = cv2.VideoCapture("test_video.mp4"))
        cap = cv2.VideoCapture(0) 

        while self._run_flag:
            ret, frame = cap.read()
            if not ret: break

            # 1. YOLO 추론
            results = self.yolo.track(frame, persist=True, verbose=False)
            result = results[0]
            
            # 시각화용 프레임
            annotated_frame = result.plot()

            # 2. 데이터 추출
            current_kpts = np.zeros((17, 2)) # 사람 없으면 0으로 채움
            
            if result.boxes and result.keypoints:
                # 가장 신뢰도 높은 사람 1명만 선택
                # (실제론 track_id 유지 로직을 넣으면 더 좋음)
                kpts = result.keypoints.data[0].cpu().numpy() # (17, 3) -> x, y, conf
                current_kpts = kpts[:, :2] # x, y만 사용

            # 3. 버퍼에 추가 & 정규화
            norm_kpts = self.normalize_keypoints(current_kpts)
            self.frame_buffer.append(norm_kpts)

            # 4. 행동 분류 (버퍼가 꽉 찼을 때만)
            pred_label = "Buffering..."
            if len(self.frame_buffer) == WINDOW_SIZE:
                # Tensor 변환: (300, 17, 2) -> (1, 2, 300, 17) (Model Input Shape)
                input_data = np.array(self.frame_buffer)
                input_data = input_data.transpose(2, 0, 1) # (C, T, V)
                input_data = np.expand_dims(input_data, axis=0) # Batch dim
                
                input_tensor = torch.FloatTensor(input_data).to(DEVICE)

                with torch.no_grad():
                    output = self.stgcn(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    score, idx = torch.max(probs, 1)
                    
                    if score.item() > 0.6: # 확신도 60% 이상일 때만 표시
                        pred_label = f"{CLASSES[idx.item()]} ({score.item()*100:.1f}%)"
                    else:
                        pred_label = "Uncertain"

            # 5. GUI 전송
            # 화면에 텍스트 그리기
            cv2.putText(annotated_frame, f"Action: {pred_label}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # OpenCV(BGR) -> PyQt(RGB) 변환
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            self.change_pixmap_signal.emit(qt_image)
            self.update_label_signal.emit(pred_label)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ================= [3. 메인 윈도우] =================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Action Recognition System")
        self.resize(1000, 800)

        # 위젯 설정
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # 비디오 표시 라벨
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setFixedSize(960, 540) # 화면 크기 조절
        self.layout.addWidget(self.image_label)

        # 결과 텍스트 라벨
        self.result_label = QLabel("Waiting for Start...", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 30px; font-weight: bold; color: blue;")
        self.layout.addWidget(self.result_label)

        # 버튼
        self.start_btn = QPushButton("Start Detection", self)
        self.start_btn.clicked.connect(self.start_video)
        self.start_btn.setFixedSize(200, 50)
        self.layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)

        # 스레드 초기화
        self.thread = None

    def start_video(self):
        if self.thread is None:
            self.thread = DetectionThread()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.update_label_signal.connect(self.update_text)
            self.thread.start()
            self.start_btn.setText("Stop")
            self.start_btn.clicked.disconnect()
            self.start_btn.clicked.connect(self.stop_video)

    def stop_video(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
            self.start_btn.setText("Start Detection")
            self.start_btn.clicked.disconnect()
            self.start_btn.clicked.connect(self.start_video)

    def update_image(self, qt_image):
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def update_text(self, text):
        self.result_label.setText(text)
        # 위험 상황(Assault, Swoon)일 때 빨간색으로 표시
        if "Assault" in text or "Swoon" in text or "Vandalism" in text:
            self.result_label.setStyleSheet("font-size: 30px; font-weight: bold; color: red;")
        elif "Normal" in text:
            self.result_label.setStyleSheet("font-size: 30px; font-weight: bold; color: green;")
        else:
            self.result_label.setStyleSheet("font-size: 30px; font-weight: bold; color: black;")

    def closeEvent(self, event):
        self.stop_video()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())