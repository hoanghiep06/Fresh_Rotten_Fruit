import sys
import time
import numpy as np
import cv2
import json
import threading
from queue import Queue
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread


# Prediction Thread
class PredictionThread(QThread):
    prediction_ready = pyqtSignal(str, str, float, float)
    no_fruit_detected = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.model = None
        self.id2fruit_label = {}
        self.freshness_id2label = {}
        self.IMG_SIZE = (128, 128)
        self.CROP_SIZE = (300, 200)

    def load_model(self):

        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model('./fruit_freshness_model.h5')

            with open('./id2fruit_label.json', 'r') as f:
                id2fruit_label = json.load(f)
            with open('./freshness_id2label.json', 'r') as f:
                freshness_id2label = json.load(f)

            self.id2fruit_label = {int(v): k for k, v in id2fruit_label.items()}
            self.freshness_id2label = {int(v): k for k, v in freshness_id2label.items()}
            return True
        except Exception as e:
            print(f"Lỗi load model: {e}")
            return False

    def add_frame(self, frame):

        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def image_processing(self, img):

        import tensorflow as tf
        img = cv2.resize(img, self.IMG_SIZE)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return tf.keras.applications.efficientnet.preprocess_input(img)

    def predict(self, img):

        if self.model is None:
            return "", "", 0.0, 0.0

        try:
            img = self.image_processing(img)
            pred_fruit, pred_freshness = self.model.predict(img, verbose=0)


            fruit_idx = np.argmax(pred_fruit)
            freshness_idx = np.argmax(pred_freshness)

            fruit_confidence = float(np.max(pred_fruit))
            freshness_confidence = float(np.max(pred_freshness))

            fruit_label = self.id2fruit_label.get(fruit_idx, "")
            freshness_label = self.freshness_id2label.get(freshness_idx, "")

            return fruit_label, freshness_label, fruit_confidence, freshness_confidence

        except Exception as e:
            print(f"Lỗi prediction: {e}")
            return "", "", 0.0, 0.0

    def run(self):

        self.running = True
        MIN_CONFIDENCE = 0.6

        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    fruit, freshness, fruit_conf, fresh_conf = self.predict(frame)


                    if fruit and freshness and fruit_conf > MIN_CONFIDENCE and fresh_conf > MIN_CONFIDENCE:
                        self.prediction_ready.emit(fruit, freshness, fruit_conf, fresh_conf)
                    elif fruit_conf <= MIN_CONFIDENCE or fresh_conf <= MIN_CONFIDENCE:

                        self.no_fruit_detected.emit()


                time.sleep(0.1)

            except Exception as e:
                print(f"Lỗi trong prediction thread: {e}")

    def stop(self):

        self.running = False
        while not self.frame_queue.empty():
            self.frame_queue.get()


# Main App
class WebcamApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fresh_Rotten_Fruit - Fixed Version")
        self.setGeometry(100, 100, 700, 600)

        # UI Components
        self.image_label = QLabel("Camera sẽ hiển thị ở đây")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray;")

        self.status_label = QLabel("Trạng thái: Đang khởi tạo...")
        self.prediction_label = QLabel("Kết quả: Chưa có dự đoán")
        self.prediction_label.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")

        self.btn_start = QPushButton("Bật Camera")
        self.btn_add_img = QPushButton("Tải ảnh lên")


        hbox_controls = QHBoxLayout()
        hbox_controls.addWidget(self.btn_add_img)
        hbox_controls.addWidget(self.btn_start)

        vbox_main = QVBoxLayout()
        vbox_main.addWidget(self.status_label)
        vbox_main.addWidget(self.image_label)
        vbox_main.addWidget(self.prediction_label)
        vbox_main.addLayout(hbox_controls)
        self.setLayout(vbox_main)


        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None

        # Image mode tracking
        self.current_mode = "none"
        self.loaded_image = None
        self.loaded_image_processed = None


        self.prediction_thread = PredictionThread()
        self.prediction_thread.prediction_ready.connect(self.on_prediction_ready)
        self.prediction_thread.no_fruit_detected.connect(self.on_no_fruit_detected)


        self.last_prediction = ("", "")
        self.last_confidence = (0.0, 0.0)
        self.prediction_time = 0
        self.last_valid_prediction_time = 0
        self.frame_count = 0

        self.btn_start.clicked.connect(self.toggle_camera)
        self.btn_add_img.clicked.connect(self.load_image)  # Thêm function load image

        self.load_model_async()

    def load_model_async(self):


        def load_worker():
            self.status_label.setText("Trạng thái: Đang tải model...")
            success = self.prediction_thread.load_model()
            if success:
                self.status_label.setText("Trạng thái: Model đã sẵn sàng - Có thể dự đoán")
            else:
                self.status_label.setText("Trạng thái: Lỗi tải model - Chỉ hiển thị camera")
                print("Không tải được model, chỉ chạy camera thôi")

        thread = threading.Thread(target=load_worker)
        thread.daemon = True
        thread.start()

    def start_camera(self):
        if self.current_mode == "image":
            self.loaded_image = None
            self.loaded_image_processed = None

        if self.cap and self.cap.isOpened():
            return

        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        camera_indices = [0, 1, 2]
        found_camera = False

        self.status_label.setText("Trạng thái: Đang tìm camera...")

        for backend in backends:
            for index in camera_indices:
                try:
                    self.cap = cv2.VideoCapture(index, backend)
                    if self.cap.isOpened():
                        # Test read một frame
                        ret, _ = self.cap.read()
                        if ret:
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            found_camera = True
                            break
                        else:
                            self.cap.release()
                except:
                    continue
            if found_camera:
                break

        if not found_camera:
            self.status_label.setText("Trạng thái: Không tìm thấy camera")
            QMessageBox.warning(self, "Lỗi", "Không thể mở webcam!")
            return


        if not self.prediction_thread.isRunning():
            self.prediction_thread.start()

        self.current_mode = "camera"
        self.timer.start(33)  # ~30 FPS
        self.btn_start.setText("Tắt Camera")
        self.status_label.setText("Trạng thái: Camera đang hoạt động")

    def stop_camera(self):

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        self.timer.stop()

        if self.prediction_thread.isRunning():
            self.prediction_thread.stop()
            self.prediction_thread.wait(1000)  # Wait max 1 second

        self.current_mode = "none"
        self.image_label.clear()
        self.image_label.setText("Camera đã tắt")
        self.btn_start.setText("Bật Camera")
        self.status_label.setText("Trạng thái: Camera đã tắt")
        self.prediction_label.setText("Kết quả: Chưa có dự đoán")

    def load_image(self):

        # Dừng camera nếu đang chạy
        if self.current_mode == "camera":
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            self.timer.stop()
            self.btn_start.setText("Bật Camera")

        # Mở dialog chọn file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if not file_path:
            return

        try:
            self.loaded_image = cv2.imread(file_path)
            if self.loaded_image is None:
                QMessageBox.warning(self, "Lỗi", "Không thể đọc file ảnh!")
                return

            h, w = self.loaded_image.shape[:2]
            target_w, target_h = 640, 480


            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)

            resized = cv2.resize(self.loaded_image, (new_w, new_h))


            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            self.loaded_image = canvas


            self.loaded_image_processed = self.add_center_rectangle(
                self.loaded_image.copy(),
                (new_w, new_h),
                color=(144, 238, 144),
                guide_text=False
            )

            self.display_image(self.loaded_image_processed)


            if not self.prediction_thread.isRunning():
                self.prediction_thread.start()

            self.predict_on_loaded_image()

            self.current_mode = "image"
            self.status_label.setText(f"Trạng thái: Đã tải ảnh - {file_path.split('/')[-1]}")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi tải ảnh: {str(e)}")

    def predict_on_loaded_image(self):
        if self.loaded_image is None or self.prediction_thread.model is None:
            return

        try:
            h, w = self.loaded_image.shape[:2]
            # crop_w, crop_h = self.prediction_thread.CROP_SIZE
            #
            # x1 = w // 2 - crop_w // 2
            # y1 = h // 2 - crop_h // 2
            # x2 = x1 + crop_w
            # y2 = y1 + crop_h

            center_crop = self.loaded_image[:, :].copy()

            fruit, freshness, fruit_conf, fresh_conf = self.prediction_thread.predict(center_crop)


            if fruit and freshness and fruit_conf > 0.6 and fresh_conf > 0.6:
                self.on_prediction_ready(fruit, freshness, fruit_conf, fresh_conf)


                img_with_text = self.loaded_image_processed.copy()
                text = f"{freshness} {fruit}"
                conf_text = f"({fruit_conf:.2f}, {fresh_conf:.2f})"

                cv2.putText(img_with_text, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(img_with_text, conf_text, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

                self.display_image(img_with_text)
            else:
                self.on_no_fruit_detected()

        except Exception as e:
            print(f"Lỗi predict trên ảnh: {e}")

    def display_image(self, image):

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def add_center_rectangle(self, frame, crop_size=(300, 200), color=(0, 255, 100), alpha=0.15, guide_text=True):
        h, w, _ = frame.shape
        # Tính toán vị trí center
        x1 = w // 2 - crop_size[0] // 2
        y1 = h // 2 - crop_size[1] // 2
        x2 = x1 + crop_size[0]
        y2 = y1 + crop_size[1]


        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # Fill rectangle
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Border

        # Thêm text hướng dẫn, xuất hiện trong webcam, không có trong upload ảnh
        if guide_text is True:
            cv2.putText(frame, "Dat trai cay vao day", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def toggle_camera(self):

        if self.cap and self.cap.isOpened():
            self.stop_camera()
        else:
            self.start_camera()

    def update_frame(self):
        if self.current_mode != "camera":
            return

        if self.cap is None or not self.cap.isOpened():
            self.stop_camera()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return


        frame_display = self.add_center_rectangle(frame.copy(), self.prediction_thread.CROP_SIZE)


        self.frame_count += 1
        if self.frame_count % 10 == 0 and self.prediction_thread.model is not None:
            h, w, _ = frame.shape
            crop_w, crop_h = self.prediction_thread.CROP_SIZE

            x1 = w // 2 - crop_w // 2
            y1 = h // 2 - crop_h // 2
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            center_crop = frame[y1:y2, x1:x2].copy()
            self.prediction_thread.add_frame(center_crop)


        current_time = time.time()
        should_show_prediction = False


        if current_time - self.prediction_time < 2.0:  # Prediction mới trong 2s
            should_show_prediction = True
        elif current_time - self.last_valid_prediction_time < 0.5:
            should_show_prediction = True

        if should_show_prediction and self.last_prediction[0] and self.last_prediction[1]:
            fruit, freshness = self.last_prediction
            fruit_conf, fresh_conf = self.last_confidence


            text = f"{freshness} {fruit}"
            conf_text = f"({fruit_conf:.2f}, {fresh_conf:.2f})"

            cv2.putText(frame_display, text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame_display, conf_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)


        self.display_image(frame_display)

    def on_prediction_ready(self, fruit, freshness, fruit_conf, fresh_conf):
        self.last_prediction = (fruit, freshness)
        self.last_confidence = (fruit_conf, fresh_conf)
        self.prediction_time = time.time()
        self.last_valid_prediction_time = time.time()

        self.prediction_label.setText(
            f"Kết quả: {freshness} {fruit} (Confidence: {fruit_conf:.2f}, {fresh_conf:.2f})"
        )

    def on_no_fruit_detected(self):
        self.prediction_label.setText("Kết quả: Không phát hiện quả rõ ràng (confidence thấp)")


    def closeEvent(self, event):
        self.stop_camera()
        self.loaded_image = None
        self.loaded_image_processed = None
        event.accept()


# Run
def main():
    app = QApplication(sys.argv)
    win = WebcamApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
