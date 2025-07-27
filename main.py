import sys
import cv2
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap, QImage, QColor

class FaceDetectionApp(QWidget):
    def __init__(self, camera_index=0):
        super().__init__()
        self.setWindowTitle("IR Camera 얼굴 인식")

        # UI 요소
        self.video_label = QLabel()
        self.status_label = QLabel("상태: 초기화 중...")
        self.status_label.setStyleSheet("font-size: 16px; color: blue")
        self.quit_button = QPushButton("종료")
        self.quit_button.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)

        # 얼굴 탐지기 초기화
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # 카메라 설정
        self.cap = cv2.VideoCapture(camera_index)  # IR 카메라 인덱스 번호
        if not self.cap.isOpened():
            self.status_label.setText("❌ 카메라 열기 실패")
            return

        # 타이머 설정 (30fps 정도)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # 약 30fps

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("❌ 프레임 읽기 실패")
            return

        # 흑백 변환 (IR 카메라는 대부분 흑백)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 탐지
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # 얼굴 영역 표시
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 상태 텍스트 업데이트
        if len(faces) > 0:
            self.status_label.setText("✅ 얼굴 인식됨")
            self.status_label.setStyleSheet("font-size: 16px; color: green")
        else:
            self.status_label.setText("❌ 얼굴 없음")
            self.status_label.setStyleSheet("font-size: 16px; color: red")

        # OpenCV → Qt 이미지 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 라벨에 출력
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.width() if self.video_label.width() > 0 else w,
            self.video_label.height() if self.video_label.height() > 0 else h,
            Qt.KeepAspectRatio
        ))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceDetectionApp(camera_index=1)  # IR 카메라 번호에 맞게 바꿔주세요 (0, 1, 2...)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
