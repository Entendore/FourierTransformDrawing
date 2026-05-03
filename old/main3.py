import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout,
                             QLabel, QSlider, QHBoxLayout)
from PyQt6.QtGui import QPainter, QPen, QColor, QMouseEvent
from PyQt6.QtCore import Qt, QTimer, QPointF
from PIL import Image, ImageDraw
import cv2

class FourierDrawing(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Transform Drawing")
        self.setGeometry(100, 100, 900, 700)

        # Drawing and Fourier data
        self.drawing = []
        self.fourier = []
        self.time = 0
        self.path = []

        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.speed = 1.0

        # UI
        main_layout = QVBoxLayout()
        self.info_label = QLabel("Draw with mouse, then press 'Compute Fourier'")
        main_layout.addWidget(self.info_label)

        btn_layout = QHBoxLayout()
        self.compute_btn = QPushButton("Compute Fourier")
        self.compute_btn.clicked.connect(self.compute_fourier)
        btn_layout.addWidget(self.compute_btn)

        self.animate_btn = QPushButton("Start Animation")
        self.animate_btn.clicked.connect(self.start_animation)
        btn_layout.addWidget(self.animate_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_drawing)
        btn_layout.addWidget(self.reset_btn)

        self.export_gif_btn = QPushButton("Export GIF")
        self.export_gif_btn.clicked.connect(self.export_gif)
        btn_layout.addWidget(self.export_gif_btn)

        self.export_mp4_btn = QPushButton("Export MP4")
        self.export_mp4_btn.clicked.connect(self.export_mp4)
        btn_layout.addWidget(self.export_mp4_btn)

        main_layout.addLayout(btn_layout)

        # Speed slider
        slider_layout = QHBoxLayout()
        self.speed_label = QLabel("Animation Speed:")
        slider_layout.addWidget(self.speed_label)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.change_speed)
        slider_layout.addWidget(self.speed_slider)
        main_layout.addLayout(slider_layout)

        self.setLayout(main_layout)

    # --- Mouse Events ---
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = [QPointF(event.position().x(), event.position().y())]
            self.path = []
            self.time = 0
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.drawing.append(QPointF(event.position().x(), event.position().y()))
            self.update()

    # --- Painting ---
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw user drawing
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        for i in range(1, len(self.drawing)):
            painter.drawLine(self.drawing[i-1], self.drawing[i])

        # Draw Fourier animation
        if self.fourier:
            x, y = self.fourier_animation(self.time, painter)
            self.path.append(QPointF(x, y))
            pen = QPen(QColor(0, 255, 100))
            pen.setWidth(2)
            painter.setPen(pen)
            for i in range(1, len(self.path)):
                painter.drawLine(self.path[i-1], self.path[i])

    # --- Fourier ---
    def compute_fourier(self):
        if not self.drawing:
            self.info_label.setText("Draw something first!")
            return
        xs = [p.x() for p in self.drawing]
        ys = [p.y() for p in self.drawing]
        cx, cy = np.mean(xs), np.mean(ys)
        points = [complex(p.x()-cx, p.y()-cy) for p in self.drawing]
        N = len(points)
        X = []
        for k in range(N):
            sum_val = sum(points[n] * np.exp(-2j * np.pi * k * n / N) for n in range(N)) / N
            X.append((abs(sum_val), sum_val, k))
        self.fourier = sorted(X, key=lambda x: x[0], reverse=True)
        self.info_label.setText("Fourier computed! Ready to animate.")

    # --- Animation ---
    def start_animation(self):
        self.path = []
        self.time = 0
        self.timer.start(16)

    def update_animation(self):
        self.time += 0.01 * self.speed
        if self.time > 1:
            self.time = 0
            self.path = []
        self.update()

    def fourier_animation(self, t, painter):
        """
        Draws rotating Fourier vectors and returns the tip of the final vector.
        """
        if not self.fourier:
            return self.width()/2, self.height()/2

        # Center of the window
        x = self.width() / 2
        y = self.height() / 2

        # Scale factor to make drawing visible
        scale = min(self.width(), self.height()) / 3

        # Draw each vector
        for amp, coeff, freq in self.fourier:
            prev_x, prev_y = x, y
            dx = scale * amp * np.cos(2 * np.pi * freq * t + np.angle(coeff))
            dy = scale * amp * np.sin(2 * np.pi * freq * t + np.angle(coeff))
            x += dx
            y += dy

            # Vector color based on frequency (low freq = warm, high freq = cool)
            hue = int(200 * (1 - min(abs(freq)/len(self.fourier), 1.0))) + 55
            pen = QPen(QColor(hue, 255-hue, 255))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(QPointF(prev_x, prev_y), QPointF(x, y))

        # Draw reconstruction path (bright green)
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        for i in range(1, len(self.path)):
            painter.drawLine(self.path[i-1], self.path[i])

        return x, y

    # --- UI Helpers ---
    def change_speed(self):
        self.speed = self.speed_slider.value()

    def reset_drawing(self):
        self.drawing = []
        self.path = []
        self.fourier = []
        self.time = 0
        self.update()
        self.info_label.setText("Draw with mouse, then press 'Compute Fourier'")

    # --- Export GIF ---
    def export_gif(self):
        if not self.fourier:
            self.info_label.setText("Compute Fourier first!")
            return
        self.info_label.setText("Exporting GIF...")
        frames = []
        N_frames = 200
        width, height = self.width(), self.height()
        scale = min(width, height) / 4

        for i in range(N_frames):
            t = i / N_frames
            img = Image.new("RGB", (width, height), (10, 10, 30))
            draw = ImageDraw.Draw(img)
            x, y = width/2, height/2
            for amp, coeff, freq in self.fourier:
                prev_x, prev_y = x, y
                dx = scale * amp * np.cos(2 * np.pi * freq * t + np.angle(coeff))
                dy = scale * amp * np.sin(2 * np.pi * freq * t + np.angle(coeff))
                x += dx
                y += dy
                hue = int(255 * min(abs(freq)/len(self.fourier), 1.0))
                draw.line([prev_x, prev_y, x, y], fill=(hue, 255-hue, 200))
            frames.append(img)
        frames[0].save("fourier_animation.gif", save_all=True, append_images=frames[1:], optimize=True, duration=50)
        self.info_label.setText("GIF saved: fourier_animation.gif")

    # --- Export MP4 ---
    def export_mp4(self):
        if not self.fourier:
            self.info_label.setText("Compute Fourier first!")
            return
        width, height = self.width(), self.height()
        scale = min(width, height) / 4
        N_frames = 200
        out = cv2.VideoWriter('fourier_animation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for i in range(N_frames):
            t = i / N_frames
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:] = [10, 10, 30]
            x, y = width/2, height/2
            for amp, coeff, freq in self.fourier:
                prev_x, prev_y = x, y
                dx = scale * amp * np.cos(2*np.pi*freq*t + np.angle(coeff))
                dy = scale * amp * np.sin(2*np.pi*freq*t + np.angle(coeff))
                x += dx
                y += dy
                hue = int(255 * min(abs(freq)/len(self.fourier), 1.0))
                color = (200, 255-hue, hue)
                cv2.line(img, (int(prev_x), int(prev_y)), (int(x), int(y)), color, 1)
            out.write(img)
        out.release()
        self.info_label.setText("MP4 saved: fourier_animation.mp4")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FourierDrawing()
    window.show()
    sys.exit(app.exec())
