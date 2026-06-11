import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QColor, QPen

from algorithms import render_epicycle_chain, render_frame_data, compute_dft
from utils import interpolate_path

class FourierCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        
        # Data Structures
        self.strokes = []
        self.current_stroke = []
        self.normalized_points = []
        
        # Numba-friendly data
        self.freqs = np.array([])
        self.amps = np.array([])
        self.phases = np.array([])
        
        # Animation State
        self.time = 0.0
        self.trail = []
        self.playing = False
        self.speed = 0.05
        self.max_terms = 100
        
        # Visual Settings
        self.show_circles = True
        self.show_vectors = True
        self.trail_color = QColor(0, 255, 128)
        self.epicycle_color = QColor(100, 100, 100, 150)
        self.background_color = QColor(30, 30, 30)
        
        # Drawing State
        self.drawing_mode = False
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16) # ~60 FPS

    def set_data(self, points):
        if not points:
            return
        
        # Centering and Scaling
        pts = np.array(points)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        pts[:, 0] -= cx
        pts[:, 1] -= cy
        
        max_dim = max(np.abs(pts[:, 0]).max(), np.abs(pts[:, 1]).max())
        if max_dim > 0:
            scale = min(self.width(), self.height()) * 0.4 / max_dim
            pts *= scale
            
        self.normalized_points = [tuple(p) for p in pts]
        
        # Compute Fourier Coefficients
        self.freqs, self.amps, self.phases = compute_dft(self.normalized_points)
        
        # Reset Animation
        self.trail = []
        self.time = 0.0
        self.playing = True

    def clear_all(self):
        self.strokes = []
        self.current_stroke = []
        self.normalized_points = []
        self.freqs = np.array([])
        self.trail = []
        self.update()

    # --- Mouse Events ---
    
    def mousePressEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton:
            pos = event.position()
            self.current_stroke = [(pos.x(), pos.y())]

    def mouseMoveEvent(self, event):
        if self.drawing_mode and (event.buttons() & Qt.LeftButton):
            pos = event.position()
            self.current_stroke.append((pos.x(), pos.y()))
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton:
            if len(self.current_stroke) > 5:
                self.strokes.append(self.current_stroke)
                self.recompute_drawing()
            self.current_stroke = []

    def recompute_drawing(self):
        all_points = []
        for stroke in self.strokes:
            all_points.extend(stroke)
            all_points.append(stroke[-1]) # duplicate point to avoid jump
        
        if len(all_points) > 10:
            sampled = interpolate_path(all_points, num_samples=1000)
            self.set_data(sampled)

    def update_animation(self):
        if self.playing and len(self.freqs) > 0:
            center_x = self.width() / 2
            center_y = self.height() / 2
            
            # Call Numba accelerated function
            x, y = render_frame_data(
                self.freqs, self.amps, self.phases, 
                self.time, self.max_terms, 
                self.width(), self.height(), 
                center_x, center_y
            )
            
            self.trail.append((x, y))
            
            if len(self.trail) > 600:
                self.trail.pop(0)
                
            self.time += self.speed
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.background_color)

        # Draw User Strokes
        if self.drawing_mode:
            pen = QPen(QColor(200, 200, 200), 2)
            painter.setPen(pen)
            for stroke in self.strokes:
                for i in range(len(stroke)-1):
                    painter.drawLine(QPointF(*stroke[i]), QPointF(*stroke[i+1]))
            if self.current_stroke:
                for i in range(len(self.current_stroke)-1):
                    painter.drawLine(QPointF(*self.current_stroke[i]), QPointF(*self.current_stroke[i+1]))
            return

        # Draw Fourier Epicycles
        if len(self.freqs) > 0:
            cx, cy = self.width() / 2, self.height() / 2
            
            # Get vector chain from Numba function
            points_chain = render_epicycle_chain(
                self.freqs, self.amps, self.phases, self.time, self.max_terms
            )
            
            # Draw Circles and Lines
            for i in range(len(points_chain) - 1):
                px, py = points_chain[i]
                x, y = points_chain[i+1]
                
                # Shift to center
                px_s, py_s = px + cx, py + cy
                x_s, y_s = x + cx, y + cy
                
                amp = self.amps[i]
                
                if self.show_circles:
                    painter.setPen(QPen(self.epicycle_color, 1))
                    painter.drawEllipse(QPointF(px_s, py_s), amp, amp)
                
                if self.show_vectors:
                    painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
                    painter.drawLine(QPointF(px_s, py_s), QPointF(x_s, y_s))

            # Draw Trail
            if len(self.trail) > 1:
                for i in range(1, len(self.trail)):
                    alpha = int(255 * (i / len(self.trail)))
                    color = QColor(self.trail_color.red(), self.trail_color.green(), self.trail_color.blue(), alpha)
                    painter.setPen(QPen(color, 2))
                    painter.drawLine(QPointF(*self.trail[i-1]), QPointF(*self.trail[i]))