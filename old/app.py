import sys
import math
import re
import random
import numpy as np
import cv2
import imageio
import numba
from numba import jit, float64
from xml.etree import ElementTree as ET

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QSlider, QComboBox, QCheckBox, QFileDialog, 
    QColorDialog, QGroupBox, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QMouseEvent

# ------------------------------------------------------------------------------------------------
# 1. ALGORITHMS & DATA STRUCTURES (Numba Accelerated)
# ------------------------------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def render_epicycle_chain(freqs, amps, phases, time_t, max_terms):
    """
    Calculates the chain of vectors for epicycles at a specific time.
    Returns an array of (x,y) points representing the joints of the vectors.
    """
    n = min(len(freqs), max_terms)
    points = np.empty((n + 1, 2), dtype=np.float64)
    
    x = 0.0
    y = 0.0
    points[0, 0] = x
    points[0, 1] = y
    
    for i in range(n):
        angle = 2.0 * np.pi * freqs[i] * time_t + phases[i]
        x += amps[i] * math.cos(angle)
        y += amps[i] * math.sin(angle)
        
        points[i+1, 0] = x
        points[i+1, 1] = y
        
    return points

@jit(nopython=True, cache=True)
def render_frame_data(freqs, amps, phases, time_t, max_terms, width, height, cx, cy):
    """
    Calculates the final drawing point (tip of the last vector) for trail generation.
    """
    x = cx
    y = cy
    
    n = min(len(freqs), max_terms)
    
    for i in range(n):
        angle = 2.0 * np.pi * freqs[i] * time_t + phases[i]
        x += amps[i] * math.cos(angle)
        y += amps[i] * math.sin(angle)
        
    return x, y

def compute_dft(points):
    """
    Computes the Discrete Fourier Transform coefficients using NumPy.
    Prepares data for Numba-accelerated rendering.
    """
    if not points:
        return np.array([]), np.array([]), np.array([])
    
    pts = np.array([complex(p[0], p[1]) for p in points])
    
    # Compute FFT
    coeffs = np.fft.fft(pts)
    freqs = np.fft.fftfreq(len(pts))
    
    # Extract magnitude and phase
    amps = np.abs(coeffs) / len(pts)
    phases = np.angle(coeffs)
    
    # Sort by amplitude descending
    indices = np.argsort(amps)[::-1]
    
    return freqs[indices], amps[indices], phases[indices]

# ------------------------------------------------------------------------------------------------
# 2. UTILITY FUNCTIONS (SVG, Math, Interpolation)
# ------------------------------------------------------------------------------------------------

def interpolate_path(points, num_samples=500):
    """Resamples a path to have uniform distance between points."""
    if len(points) < 2:
        return points
    
    points = np.array(points)
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    
    if cumulative_dist[-1] == 0:
        return points
    
    uniform_dist = np.linspace(0, cumulative_dist[-1], num_samples)
    
    new_x = np.interp(uniform_dist, cumulative_dist, points[:, 0])
    new_y = np.interp(uniform_dist, cumulative_dist, points[:, 1])
    
    return list(zip(new_x, new_y))

def parse_svg_path(d_attr):
    """Parses SVG path data string into points."""
    tokens = re.findall(r'([MLCQAZmlcqaz])([^MLCQAZmlcqaz]*)', d_attr)
    points = []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    
    def to_floats(s):
        return [float(x) for x in re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', s)]

    for cmd, args in tokens:
        vals = to_floats(args)
        cmdu = cmd.upper()
        
        if cmdu == 'M':
            for i in range(0, len(vals), 2):
                cur = (vals[i], vals[i+1])
                start = cur
                points.append(cur)
        elif cmdu == 'L':
            for i in range(0, len(vals), 2):
                nxt = (vals[i], vals[i+1])
                points.append(nxt)
                cur = nxt
        elif cmdu == 'C':
            for i in range(0, len(vals), 6):
                p1 = (vals[i], vals[i+1])
                p2 = (vals[i+2], vals[i+3])
                p3 = (vals[i+4], vals[i+5])
                # Sample bezier
                for t in np.linspace(0, 1, 20):
                    x = (1-t)**3*cur[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
                    y = (1-t)**3*cur[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
                    points.append((x,y))
                cur = p3
        elif cmdu == 'Q':
            for i in range(0, len(vals), 4):
                p1 = (vals[i], vals[i+1])
                p2 = (vals[i+2], vals[i+3])
                for t in np.linspace(0, 1, 20):
                    x = (1-t)**2*cur[0] + 2*(1-t)*t*p1[0] + t**2*p2[0]
                    y = (1-t)**2*cur[1] + 2*(1-t)*t*p1[1] + t**2*p2[1]
                    points.append((x,y))
                cur = p2
        elif cmdu == 'Z':
            points.append(start)
            cur = start
            
    return points

# ------------------------------------------------------------------------------------------------
# 3. CANVAS WIDGET
# ------------------------------------------------------------------------------------------------

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

    # --- Mouse Events (PySide6 specific: event.position() returns QPointF) ---
    
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

# ------------------------------------------------------------------------------------------------
# 4. MAIN WINDOW & EXPORT (CV2 Only, No Pillow)
# ------------------------------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified Fourier Studio (PySide6 + Numba)")
        self.resize(1200, 800)
        
        self.canvas = FourierCanvas()
        self.setCentralWidget(self.canvas)
        self.setup_control_panel()
        
        self.presets = [
            {"name": "Circle", "x": "cos(t)", "y": "sin(t)"},
            {"name": "Rose", "x": "cos(2*t)*cos(t)", "y": "cos(2*t)*sin(t)"},
            {"name": "Lissajous", "x": "sin(3*t)", "y": "sin(4*t)"},
            {"name": "Star", "x": "cos(t) + 0.5*cos(5*t)", "y": "sin(t) + 0.5*sin(5*t)"},
        ]
        self.populate_presets()

    def setup_control_panel(self):
        dock = QWidget()
        layout = QVBoxLayout(dock)
        
        # Mode Group
        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout()
        self.draw_btn = QPushButton("Enter Drawing Mode")
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self.toggle_drawing_mode)
        mode_layout.addWidget(self.draw_btn)
        clear_btn = QPushButton("Clear Canvas")
        clear_btn.clicked.connect(self.canvas.clear_all)
        mode_layout.addWidget(clear_btn)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Input Group
        input_group = QGroupBox("Import / Presets")
        input_layout = QVBoxLayout()
        svg_btn = QPushButton("Import SVG")
        svg_btn.clicked.connect(self.import_svg)
        input_layout.addWidget(svg_btn)
        self.preset_combo = QComboBox()
        input_layout.addWidget(self.preset_combo)
        load_preset_btn = QPushButton("Load Preset")
        load_preset_btn.clicked.connect(self.load_preset)
        input_layout.addWidget(load_preset_btn)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Visuals Group
        vis_group = QGroupBox("Visual Settings")
        vis_layout = QVBoxLayout()
        
        term_layout = QHBoxLayout()
        term_layout.addWidget(QLabel("Detail:"))
        self.term_slider = QSlider(Qt.Horizontal)
        self.term_slider.setRange(1, 300)
        self.term_slider.setValue(100)
        self.term_slider.valueChanged.connect(lambda v: setattr(self.canvas, 'max_terms', v))
        term_layout.addWidget(self.term_slider)
        vis_layout.addLayout(term_layout)
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(lambda v: setattr(self.canvas, 'speed', v / 100.0))
        speed_layout.addWidget(self.speed_slider)
        vis_layout.addLayout(speed_layout)
        
        self.circles_chk = QCheckBox("Show Circles")
        self.circles_chk.setChecked(True)
        self.circles_chk.toggled.connect(lambda s: setattr(self.canvas, 'show_circles', s))
        vis_layout.addWidget(self.circles_chk)
        
        color_btn = QPushButton("Trail Color")
        color_btn.clicked.connect(self.pick_color)
        vis_layout.addWidget(color_btn)
        
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        # Export Group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        export_gif_btn = QPushButton("Export GIF")
        export_gif_btn.clicked.connect(self.export_gif)
        export_layout.addWidget(export_gif_btn)
        export_mp4_btn = QPushButton("Export MP4")
        export_mp4_btn.clicked.connect(self.export_mp4)
        export_layout.addWidget(export_mp4_btn)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        layout.addStretch()
        
        dock_widget = QWidget()
        dock_widget.setLayout(layout)
        dock_widget.setFixedWidth(250)
        
        main_layout = QHBoxLayout()
        main_layout.addWidget(dock_widget)
        main_layout.addWidget(self.canvas, 1)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def populate_presets(self):
        for p in self.presets:
            self.preset_combo.addItem(p["name"])

    def toggle_drawing_mode(self, checked):
        self.canvas.drawing_mode = checked
        self.canvas.playing = not checked
        self.draw_btn.setText("Exit Drawing Mode" if checked else "Enter Drawing Mode")
        if checked:
            self.canvas.clear_all()
            self.canvas.update()

    def import_svg(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open SVG", "", "SVG Files (*.svg)")
        if fname:
            try:
                tree = ET.parse(fname)
                root = tree.getroot()
                for elem in root.iter():
                    if elem.tag.endswith('path'):
                        d = elem.attrib.get('d')
                        if d:
                            pts = parse_svg_path(d)
                            pts = interpolate_path(pts, num_samples=800)
                            self.canvas.set_data(pts)
                            break
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to parse SVG: {e}")

    def load_preset(self):
        idx = self.preset_combo.currentIndex()
        preset = self.presets[idx]
        t_vals = np.linspace(0, 2 * math.pi, 500)
        points = []
        for t in t_vals:
            try:
                x = eval(preset['x'], {"math": math, "t": t})
                y = eval(preset['y'], {"math": math, "t": t})
                points.append((x * 200, y * 200))
            except:
                continue
        self.canvas.drawing_mode = False
        self.draw_btn.setChecked(False)
        self.canvas.set_data(points)

    def pick_color(self):
        c = QColorDialog.getColor(self.canvas.trail_color, self)
        if c.isValid():
            self.canvas.trail_color = c

    def export_gif(self):
        if len(self.canvas.freqs) == 0:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save GIF", "", "GIF Files (*.gif)")
        if fname:
            self.run_export(fname, type='gif')

    def export_mp4(self):
        if len(self.canvas.freqs) == 0:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save MP4", "", "MP4 Files (*.mp4)")
        if fname:
            self.run_export(fname, type='mp4')

    def run_export(self, filename, type='gif'):
        self.canvas.playing = False
        
        progress = QProgressBar(self)
        progress.setRange(0, 100)
        progress.setWindowTitle("Rendering...")
        progress.show()
        
        width, height = self.canvas.width(), self.canvas.height()
        total_frames = 100
        trail = []
        max_trail = 200
        
        # Numba accelerated data
        freqs = self.canvas.freqs
        amps = self.canvas.amps
        phases = self.canvas.phases
        max_terms = self.canvas.max_terms
        speed = self.canvas.speed
        cx, cy = width/2, height/2
        
        frames = []
        
        # Create color tuples for CV2 (BGR)
        trail_color_bgr = (
            self.canvas.trail_color.blue(), 
            self.canvas.trail_color.green(), 
            self.canvas.trail_color.red()
        )
        
        for i in range(total_frames):
            progress.setValue(int((i/total_frames)*100))
            QApplication.processEvents()
            
            time_t = speed * i
            
            # Create Black Image (numpy array)
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Calculate tip position using Numba
            x, y = render_frame_data(freqs, amps, phases, time_t, max_terms, width, height, cx, cy)
            
            trail.append((int(x), int(y)))
            if len(trail) > max_trail:
                trail.pop(0)
            
            # Draw Trail using CV2
            if len(trail) > 1:
                for k in range(1, len(trail)):
                    cv2.line(img, trail[k-1], trail[k], trail_color_bgr, 2)
            
            frames.append(img)
            
        if type == 'gif':
            # imageio writes RGB, CV2 used BGR. Convert.
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            imageio.mimsave(filename, frames_rgb, fps=30)
        else:
            # MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
            for f in frames:
                out.write(f)
            out.release()
            
        progress.close()
        self.canvas.playing = True

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())