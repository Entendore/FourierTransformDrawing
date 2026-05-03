import sys
import math
import numpy as np
import cv2
import imageio
from xml.etree import ElementTree as ET

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QSlider, QComboBox, QCheckBox, QFileDialog, 
    QColorDialog, QGroupBox, QProgressBar, QMessageBox, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from canvas_widget import FourierCanvas
from utils import parse_svg_path, interpolate_path
from algorithms import render_frame_data

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified Fourier Studio (PySide6 + Numba)")
        self.resize(1200, 800)
        
        self.canvas = FourierCanvas()
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