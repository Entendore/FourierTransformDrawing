#!/usr/bin/env python3
"""
Unified Fourier Studio — Pro Edition (Layers & Advanced Settings)
Single-file PySide6 + Numba application.
"""

import sys
import math
import re
import json
import os
import copy

import numpy as np
from numba import jit

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QCheckBox, QFileDialog,
    QColorDialog, QGroupBox, QProgressBar, QMessageBox, QStatusBar,
    QSpinBox, QDoubleSpinBox, QSizePolicy, QListWidget,
    QListWidgetItem, QLineEdit, QFormLayout, QFrame, QSplitter
)
from PySide6.QtCore import Qt, QTimer, QPointF, QPoint
from PySide6.QtGui import (
    QPainter, QColor, QPen, QKeySequence, QTransform, QImage,
    QRadialGradient, QPainterPath, QIcon, QShortcut
)

import cv2
import imageio
from xml.etree import ElementTree as ET

# ─────────────────────────────────────────────
#  Numba-accelerated algorithms
# ─────────────────────────────────────────────

@jit(nopython=True, cache=True)
def render_epicycle_chain(freqs, amps, phases, time_t, max_terms, cx, cy):
    n = min(len(freqs), max_terms)
    points = np.empty((n + 1, 2), dtype=np.float64)
    x, y = cx, cy
    points[0, 0] = x
    points[0, 1] = y
    for i in range(n):
        angle = 2.0 * np.pi * freqs[i] * time_t + phases[i]
        x += amps[i] * math.cos(angle)
        y += amps[i] * math.sin(angle)
        points[i + 1, 0] = x
        points[i + 1, 1] = y
    return points

@jit(nopython=True, cache=True)
def render_frame_tip(freqs, amps, phases, time_t, max_terms, cx, cy):
    x, y = cx, cy
    n = min(len(freqs), max_terms)
    for i in range(n):
        angle = 2.0 * np.pi * freqs[i] * time_t + phases[i]
        x += amps[i] * math.cos(angle)
        y += amps[i] * math.sin(angle)
    return x, y

def compute_dft(points):
    if len(points) < 2:
        return np.array([]), np.array([]), np.array([])
    pts = np.array([complex(p[0], p[1]) for p in points], dtype=complex)
    coeffs = np.fft.fft(pts)
    freqs = np.fft.fftfreq(len(pts))
    amps = np.abs(coeffs) / len(pts)
    phases = np.angle(coeffs)
    indices = np.argsort(amps)[::-1]
    return freqs[indices].astype(np.float64), amps[indices], phases[indices]

# ─────────────────────────────────────────────
#  Utilities & SVG Parsing
# ─────────────────────────────────────────────

def interpolate_path(points, num_samples=500):
    if len(points) < 2: return list(points)
    points = np.asarray(points, dtype=np.float64)
    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative = np.insert(np.cumsum(dists), 0, 0.0)
    if cumulative[-1] == 0: return list(points)
    uniform = np.linspace(0, cumulative[-1], num_samples)
    new_x = np.interp(uniform, cumulative, points[:, 0])
    new_y = np.interp(uniform, cumulative, points[:, 1])
    return list(zip(new_x.tolist(), new_y.tolist()))

def smooth_points(points, iterations=2):
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        if len(pts) < 3: break
        smoothed = pts.copy()
        smoothed[1:-1] = (pts[:-2] + pts[1:-1] + pts[2:]) / 3.0
        pts = smoothed
    return pts.tolist()

_NUM_RE = re.compile(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?')
_CMD_RE = re.compile(r'([MLCQAZHVmlcqazhv])([^MLCQAZHVmlcqazhv]*)')

def _parse_transform(transform_str):
    if not transform_str: return np.eye(3, dtype=np.float64)
    matrix = np.eye(3, dtype=np.float64)
    for m in re.finditer(r'(\w+)\s*\(([^)]*)\)', transform_str):
        name = m.group(1)
        vals = [float(v) for v in _NUM_RE.findall(m.group(2))]
        t = np.eye(3, dtype=np.float64)
        if name == 'translate':
            t[0, 2] = vals[0] if len(vals) > 0 else 0
            t[1, 2] = vals[1] if len(vals) > 1 else 0
        elif name == 'scale':
            t[0, 0] = vals[0] if len(vals) > 0 else 1
            t[1, 1] = vals[1] if len(vals) > 1 else vals[0]
        elif name == 'rotate':
            angle = math.radians(vals[0])
            c, s = math.cos(angle), math.sin(angle)
            t[0, 0], t[0, 1] = c, -s
            t[1, 0], t[1, 1] = s, c
        elif name == 'matrix' and len(vals) == 6:
            t = np.array([[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]], [0,0,1]], dtype=np.float64)
        matrix = matrix @ t
    return matrix

def _apply_transform(p, mat):
    v = np.array([p[0], p[1], 1.0])
    r = mat @ v
    return (r[0], r[1])

def parse_svg_path(d_attr, transform_mat=None):
    if transform_mat is None: transform_mat = np.eye(3, dtype=np.float64)
    strokes = []
    current = []
    cur, start = (0.0, 0.0), (0.0, 0.0)
    last_ctrl = None
    for cmd, args in _CMD_RE.findall(d_attr):
        vals = [float(x) for x in _NUM_RE.findall(args)]
        cmdu = cmd.upper()
        relative = cmd.islower()
        def rel(p): return (cur[0] + p[0], cur[1] + p[1]) if relative else p
        
        if cmdu == 'M':
            if current: strokes.append(current); current = []
            for i in range(0, len(vals) - 1, 2):
                p = rel((vals[i], vals[i+1])); cur = p; start = p
                current.append(_apply_transform(p, transform_mat))
        elif cmdu == 'L':
            for i in range(0, len(vals) - 1, 2):
                p = rel((vals[i], vals[i+1])); cur = p
                current.append(_apply_transform(p, transform_mat))
        elif cmdu == 'C':
            for i in range(0, len(vals) - 1, 6):
                p1, p2, p3 = rel((vals[i], vals[i+1])), rel((vals[i+2], vals[i+3])), rel((vals[i+4], vals[i+5]))
                for t in np.linspace(0, 1, 20):
                    mt = 1 - t
                    x = mt**3 * cur[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
                    y = mt**3 * cur[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
                    current.append(_apply_transform((x, y), transform_mat))
                cur, last_ctrl = p3, p2
        elif cmdu == 'Q':
            for i in range(0, len(vals) - 1, 4):
                p1, p2 = rel((vals[i], vals[i+1])), rel((vals[i+2], vals[i+3]))
                for t in np.linspace(0, 1, 20):
                    mt = 1 - t
                    x = mt**2 * cur[0] + 2*mt*t*p1[0] + t**2*p2[0]
                    y = mt**2 * cur[1] + 2*mt*t*p1[1] + t**2*p2[1]
                    current.append(_apply_transform((x, y), transform_mat))
                cur, last_ctrl = p2, p1
        elif cmdu == 'Z':
            if current:
                current.append(_apply_transform(start, transform_mat))
                strokes.append(current); current = []
            cur = start
    if current: strokes.append(current)
    return strokes

def parse_svg_file(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    strokes = []
    def walk(elem, parent_transform):
        t_str = elem.attrib.get('transform', '')
        local_t = _parse_transform(t_str)
        combined = parent_transform @ local_t
        tag = elem.tag.split('}')[-1]
        if tag == 'path':
            d = elem.attrib.get('d', '')
            if d: strokes.extend(parse_svg_path(d, combined))
        for child in elem: walk(child, combined)
    walk(root, np.eye(3, dtype=np.float64))
    return strokes

# ─────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────

PRESETS = [
    {"name": "Circle",       "fn": lambda t: (math.cos(t), math.sin(t))},
    {"name": "Rose",         "fn": lambda t: (math.cos(2*t)*math.cos(t), math.cos(2*t)*math.sin(t))},
    {"name": "Lissajous",    "fn": lambda t: (math.sin(3*t), math.sin(4*t))},
    {"name": "Star",         "fn": lambda t: (math.cos(t) + 0.5*math.cos(5*t), math.sin(t) + 0.5*math.sin(5*t))},
    {"name": "Heart",        "fn": lambda t: (16*math.sin(t)**3 / 16, (13*math.cos(t) - 5*math.cos(2*t) - 2*math.cos(3*t) - math.cos(4*t)) / 16)},
    {"name": "Butterfly",    "fn": lambda t: (math.sin(t) * (math.exp(math.cos(t)) - 2*math.cos(4*t) - math.sin(t/12)**5), math.cos(t) * (math.exp(math.cos(t)) - 2*math.cos(4*t) - math.sin(t/12)**5))},
    {"name": "Infinity",     "fn": lambda t: (math.cos(t) / (1 + math.sin(t)**2), math.sin(t) * math.cos(t) / (1 + math.sin(t)**2))},
]

class FourierLayer:
    def __init__(self, name="Layer"):
        self.name = name
        self.visible = True
        self.points = []           # Raw world points
        self.centroid = (0.0, 0.0)
        self.freqs = np.array([])
        self.amps = np.array([])
        self.phases = np.array([])
        self.trail = []
        
        # Per-layer settings
        self.color = QColor(0, 255, 128)
        self.max_terms = 100
        self.trail_length = 400
        self.stroke_width = 2.0
        self.show_circles = True
        self.show_vectors = True
        self.show_glow = False

    def compute_dft(self):
        if len(self.points) < 2: return
        arr = np.array(self.points, dtype=np.float64)
        self.centroid = (arr[:, 0].mean(), arr[:, 1].mean())
        
        # Center points for DFT
        centered = arr - np.array(self.centroid)
        # Flip Y for screen coordinates
        centered[:, 1] = -centered[:, 1]
        
        sampled = interpolate_path(centered.tolist(), num_samples=400)
        sampled = smooth_points(sampled, iterations=1)
        self.freqs, self.amps, self.phases = compute_dft(sampled)
        self.trail.clear()


# ─────────────────────────────────────────────
#  Fourier Canvas Widget
# ─────────────────────────────────────────────

class FourierCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.layers = []
        self.current_stroke = []

        # Global Animation State
        self.time = 0.0
        self.playing = True
        self.reverse = False
        self.speed = 0.05

        # Global Visuals
        self.background_color = QColor(30, 30, 30)
        self.show_grid = False

        # Drawing mode
        self.drawing_mode = False

        # View transform
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._panning = False
        self._last_pan = QPoint()

        # FPS
        self._frame_count = 0
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)
        self.fps = 0.0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

    def world_to_screen(self, x, y):
        return (x * self.zoom + self.pan_x + self.width() / 2,
                y * self.zoom + self.pan_y + self.height() / 2)

    def screen_to_world(self, x, y):
        return ((x - self.pan_x - self.width() / 2) / self.zoom,
                (y - self.pan_y - self.height() / 2) / self.zoom)

    def add_layer_from_points(self, points, name="Stroke"):
        layer = FourierLayer(name=name)
        layer.points = points
        layer.compute_dft()
        self.layers.append(layer)
        self.update()

    def clear_all(self):
        self.layers.clear()
        self.current_stroke.clear()
        self.time = 0.0
        self.update()

    def reset_time(self):
        self.time = 0.0
        for layer in self.layers: layer.trail.clear()
        self.update()

    # --- Mouse Events ---

    def mousePressEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton:
            pos = event.position()
            wx, wy = self.screen_to_world(pos.x(), pos.y())
            self.current_stroke = [(wx, wy)]
        elif event.button() == Qt.MiddleButton:
            self._panning = True
            self._last_pan = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.drawing_mode and (event.buttons() & Qt.LeftButton):
            pos = event.position()
            wx, wy = self.screen_to_world(pos.x(), pos.y())
            self.current_stroke.append((wx, wy))
            self.update()
        elif self._panning:
            delta = event.position().toPoint() - self._last_pan
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self._last_pan = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton:
            if len(self.current_stroke) > 5:
                self.add_layer_from_points(list(self.current_stroke), name=f"Stroke {len(self.layers)+1}")
            self.current_stroke = []
        elif event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        factor = 1.15 ** delta
        pos = event.position()
        wx_before, wy_before = self.screen_to_world(pos.x(), pos.y())
        self.zoom = max(0.1, min(self.zoom * factor, 20.0))
        wx_after, wy_after = self.screen_to_world(pos.x(), pos.y())
        self.pan_x += (wx_after - wx_before) * self.zoom
        self.pan_y += (wy_after - wy_before) * self.zoom
        self.update()

    # --- Animation ---

    def update_animation(self):
        if self.playing and self.layers:
            for layer in self.layers:
                if not layer.visible or len(layer.freqs) == 0: continue
                x, y = render_frame_tip(
                    layer.freqs, layer.amps, layer.phases,
                    self.time, layer.max_terms,
                    layer.centroid[0], -layer.centroid[1] # Flip Y back
                )
                layer.trail.append((x, y))
                if len(layer.trail) > layer.trail_length:
                    layer.trail.pop(0)
            self.time += self.speed * (-1 if self.reverse else 1)
            self.update()

    def _update_fps(self):
        self.fps = self._frame_count
        self._frame_count = 0

    # --- Paint ---

    def paintEvent(self, event):
        self._frame_count += 1
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.background_color)

        painter.translate(self.width() / 2 + self.pan_x, self.height() / 2 + self.pan_y)
        painter.scale(self.zoom, self.zoom)

        # Grid
        if self.show_grid:
            grid_size = 50
            pen = QPen(QColor(255, 255, 255, 20), 1 / self.zoom)
            painter.setPen(pen)
            for x in range(-10, 11):
                painter.drawLine(QPointF(x * grid_size, -1000), QPointF(x * grid_size, 1000))
            for y in range(-10, 11):
                painter.drawLine(QPointF(-1000, y * grid_size), QPointF(1000, y * grid_size))

        if self.drawing_mode:
            pen = QPen(QColor(200, 200, 200), 2 / self.zoom)
            painter.setPen(pen)
            for stroke in [l.points for l in self.layers]:
                for i in range(len(stroke) - 1):
                    painter.drawLine(QPointF(*stroke[i]), QPointF(*stroke[i + 1]))
            if self.current_stroke:
                for i in range(len(self.current_stroke) - 1):
                    painter.drawLine(QPointF(*self.current_stroke[i]), QPointF(*self.current_stroke[i + 1]))
            return

        for layer in self.layers:
            if not layer.visible or len(layer.freqs) == 0: continue

            cx, cy = layer.centroid[0], -layer.centroid[1]
            chain = render_epicycle_chain(
                layer.freqs, layer.amps, layer.phases,
                self.time, layer.max_terms, cx, cy
            )

            for i in range(len(chain) - 1):
                px, py = chain[i]
                x, y = chain[i + 1]
                amp = layer.amps[i]

                if layer.show_circles and amp > 0.5:
                    painter.setPen(QPen(QColor(100, 100, 100, 120), 1 / self.zoom))
                    painter.drawEllipse(QPointF(px, py), amp, amp)

                if layer.show_vectors:
                    painter.setPen(QPen(QColor(255, 255, 255, 100), 1 / self.zoom))
                    painter.drawLine(QPointF(px, py), QPointF(x, y))

            if len(layer.trail) > 1:
                trail = layer.trail
                if layer.show_glow:
                    glow_pen = QPen(QColor(layer.color.red(), layer.color.green(), layer.color.blue(), 40), (layer.stroke_width * 4) / self.zoom)
                    painter.setPen(glow_pen)
                    path = QPainterPath(); path.moveTo(QPointF(*trail[0]))
                    for p in trail[1:]: path.lineTo(QPointF(*p))
                    painter.drawPath(path)

                pen = QPen(layer.color, layer.stroke_width / self.zoom)
                painter.setPen(pen)
                path = QPainterPath(); path.moveTo(QPointF(*trail[0]))
                for p in trail[1:]: path.lineTo(QPointF(*p))
                painter.drawPath(path)

    def render_to_image(self, width=None, height=None, include_epicycles=True, transparent=False):
        if width is None: width = self.width()
        if height is None: height = self.height()

        img = QImage(width, height, QImage.Format_ARGB32)
        img.fill(Qt.transparent if transparent else self.background_color)

        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(width / 2, height / 2)
        scale_factor = min(width, height) / min(self.width(), self.height()) * self.zoom
        painter.scale(scale_factor, scale_factor)

        for layer in self.layers:
            if not layer.visible or len(layer.freqs) == 0: continue
            cx, cy = layer.centroid[0], -layer.centroid[1]

            if include_epicycles:
                chain = render_epicycle_chain(layer.freqs, layer.amps, layer.phases, self.time, layer.max_terms, cx, cy)
                for i in range(len(chain) - 1):
                    px, py = chain[i]; x, y = chain[i + 1]
                    amp = layer.amps[i]
                    if layer.show_circles and amp > 0.5:
                        painter.setPen(QPen(QColor(100, 100, 100, 120), 1 / scale_factor))
                        painter.drawEllipse(QPointF(px, py), amp, amp)
                    if layer.show_vectors:
                        painter.setPen(QPen(QColor(255, 255, 255, 100), 1 / scale_factor))
                        painter.drawLine(QPointF(px, py), QPointF(x, y))

            if len(layer.trail) > 1:
                trail = layer.trail
                pen = QPen(layer.color, layer.stroke_width / scale_factor)
                painter.setPen(pen)
                path = QPainterPath(); path.moveTo(QPointF(*trail[0]))
                for p in trail[1:]: path.lineTo(QPointF(*p))
                painter.drawPath(path)

        painter.end()
        return img


# ─────────────────────────────────────────────
#  Layer Properties Widget
# ─────────────────────────────────────────────

class LayerPropertiesWidget(QWidget):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.active_layer = None
        self._building_ui = False

        layout = QFormLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self._apply_properties)
        layout.addRow("Name:", self.name_edit)

        self.color_btn = QPushButton("Change Color")
        self.color_btn.clicked.connect(self._pick_color)
        layout.addRow(self.color_btn)

        self.terms_slider = QSlider(Qt.Horizontal)
        self.terms_slider.setRange(1, 500)
        self.terms_slider.valueChanged.connect(self._apply_properties)
        layout.addRow("Detail:", self.terms_slider)

        self.trail_slider = QSlider(Qt.Horizontal)
        self.trail_slider.setRange(10, 2000)
        self.trail_slider.valueChanged.connect(self._apply_properties)
        layout.addRow("Trail:", self.trail_slider)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.5, 10.0)
        self.width_spin.setSingleStep(0.5)
        self.width_spin.valueChanged.connect(self._apply_properties)
        layout.addRow("Width:", self.width_spin)

        self.circles_chk = QCheckBox("Circles")
        self.circles_chk.toggled.connect(self._apply_properties)
        layout.addRow(self.circles_chk)

        self.vectors_chk = QCheckBox("Vectors")
        self.vectors_chk.toggled.connect(self._apply_properties)
        layout.addRow(self.vectors_chk)

        self.glow_chk = QCheckBox("Glow")
        self.glow_chk.toggled.connect(self._apply_properties)
        layout.addRow(self.glow_chk)
        
        self.setEnabled(False)

    def load_layer(self, layer: FourierLayer):
        self.active_layer = layer
        if not layer:
            self.setEnabled(False)
            return
        self._building_ui = True
        self.setEnabled(True)
        self.name_edit.setText(layer.name)
        self.color_btn.setStyleSheet(f"background-color: {layer.color.name()}; color: white;")
        self.terms_slider.setValue(layer.max_terms)
        self.trail_slider.setValue(layer.trail_length)
        self.width_spin.setValue(layer.stroke_width)
        self.circles_chk.setChecked(layer.show_circles)
        self.vectors_chk.setChecked(layer.show_vectors)
        self.glow_chk.setChecked(layer.show_glow)
        self._building_ui = False

    def _pick_color(self):
        if not self.active_layer: return
        c = QColorDialog.getColor(self.active_layer.color, self)
        if c.isValid():
            self.active_layer.color = c
            self.color_btn.setStyleSheet(f"background-color: {c.name()}; color: white;")
            self.canvas.update()

    def _apply_properties(self):
        if self._building_ui or not self.active_layer: return
        self.active_layer.name = self.name_edit.text()
        self.active_layer.max_terms = self.terms_slider.value()
        self.active_layer.trail_length = self.trail_slider.value()
        self.active_layer.stroke_width = self.width_spin.value()
        self.active_layer.show_circles = self.circles_chk.isChecked()
        self.active_layer.show_vectors = self.vectors_chk.isChecked()
        self.active_layer.show_glow = self.glow_chk.isChecked()
        self.canvas.update()


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified Fourier Studio — Pro Edition")
        self.resize(1400, 900)

        self.canvas = FourierCanvas()
        self.layer_props = LayerPropertiesWidget(self.canvas)
        self._build_ui()
        self._build_shortcuts()
        self._build_statusbar()
        self._populate_presets()

    def _build_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left Sidebar (Global)
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("QWidget { background-color: #2b2b2b; } "
                              "QGroupBox { color: #ddd; font-weight: bold; border: 1px solid #555; margin-top: 8px; padding: 6px; } "
                              "QGroupBox::title { subcontrol-origin: margin; left: 8px; } "
                              "QPushButton { background-color: #505050; color: white; padding: 6px; border: 1px solid #666; border-radius: 3px; } "
                              "QPushButton:hover { background-color: #606060; } "
                              "QPushButton:checked { background-color: #2a8a2a; } "
                              "QLabel { color: #ddd; } QCheckBox { color: #ddd; } "
                              "QComboBox { background-color: #505050; color: white; padding: 4px; } "
                              "QSlider::groove:horizontal { height: 6px; background: #555; } "
                              "QSlider::handle:horizontal { background: #4CAF50; width: 14px; margin: -4px 0; border-radius: 7px; }")
        sl = QVBoxLayout(sidebar)

        # Mode Group
        mode_group = QGroupBox("Mode")
        ml = QVBoxLayout()
        self.draw_btn = QPushButton("Enter Drawing Mode")
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self.toggle_drawing_mode)
        ml.addWidget(self.draw_btn)
        clear_btn = QPushButton("Clear All Layers")
        clear_btn.clicked.connect(self.canvas.clear_all)
        ml.addWidget(clear_btn)
        mode_group.setLayout(ml)
        sl.addWidget(mode_group)

        # Playback Group
        pb_group = QGroupBox("Playback")
        pbl = QVBoxLayout()
        pb_row = QHBoxLayout()
        self.play_btn = QPushButton("⏸ Pause")
        self.play_btn.clicked.connect(self.toggle_play)
        pb_row.addWidget(self.play_btn)
        reset_btn = QPushButton("⟲ Reset")
        reset_btn.clicked.connect(self.canvas.reset_time)
        pb_row.addWidget(reset_btn)
        pbl.addLayout(pb_row)

        self.reverse_chk = QCheckBox("Reverse Direction")
        self.reverse_chk.toggled.connect(lambda s: setattr(self.canvas, 'reverse', s))
        pbl.addWidget(self.reverse_chk)

        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.valueChanged.connect(self._on_time_scrub)
        time_row.addWidget(self.time_slider)
        pbl.addLayout(time_row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 500)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(lambda v: setattr(self.canvas, 'speed', v / 1000.0))
        speed_row.addWidget(self.speed_slider)
        pbl.addLayout(speed_row)
        pb_group.setLayout(pbl)
        sl.addWidget(pb_group)

        # Global View Group
        view_group = QGroupBox("View & Canvas")
        vl = QVBoxLayout()
        self.grid_chk = QCheckBox("Show Grid")
        self.grid_chk.toggled.connect(lambda s: (setattr(self.canvas, 'show_grid', s), self.canvas.update()))
        vl.addWidget(self.grid_chk)
        
        bg_btn = QPushButton("Change Background Color")
        bg_btn.clicked.connect(self.pick_bg_color)
        vl.addWidget(bg_btn)

        fit_btn = QPushButton("Fit View (Reset Zoom)")
        fit_btn.clicked.connect(self.fit_view)
        vl.addWidget(fit_btn)
        view_group.setLayout(vl)
        sl.addWidget(view_group)

        # Input Group
        input_group = QGroupBox("Import / Presets")
        il = QVBoxLayout()
        svg_btn = QPushButton("Import SVG (as Layers)")
        svg_btn.clicked.connect(self.import_svg)
        il.addWidget(svg_btn)
        self.preset_combo = QComboBox()
        il.addWidget(self.preset_combo)
        load_preset_btn = QPushButton("Load Preset (New Layer)")
        load_preset_btn.clicked.connect(self.load_preset)
        il.addWidget(load_preset_btn)
        input_group.setLayout(il)
        sl.addWidget(input_group)

        # Export Group
        export_group = QGroupBox("Export")
        el = QVBoxLayout()
        self.export_epicycles_chk = QCheckBox("Include Epicycles")
        el.addWidget(self.export_epicycles_chk)
        
        self.export_transparent_chk = QCheckBox("Transparent BG (PNG only)")
        el.addWidget(self.export_transparent_chk)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(15, 60)
        self.fps_spin.setValue(30)
        fps_row.addWidget(self.fps_spin)
        el.addLayout(fps_row)

        frames_row = QHBoxLayout()
        frames_row.addWidget(QLabel("Frames:"))
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(30, 600)
        self.frames_spin.setValue(120)
        frames_row.addWidget(self.frames_spin)
        el.addLayout(frames_row)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Res:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(["Canvas", "720p", "1080p", "Square 720"])
        res_row.addWidget(self.res_combo)
        el.addLayout(res_row)

        export_row = QHBoxLayout()
        png_btn = QPushButton("PNG")
        png_btn.clicked.connect(self.export_png)
        export_row.addWidget(png_btn)
        gif_btn = QPushButton("GIF")
        gif_btn.clicked.connect(self.export_gif)
        export_row.addWidget(gif_btn)
        mp4_btn = QPushButton("MP4")
        mp4_btn.clicked.connect(self.export_mp4)
        export_row.addWidget(mp4_btn)
        el.addLayout(export_row)
        export_group.setLayout(el)
        
        # BUG FIX: Changed from sl.addWidget(export_row) to sl.addWidget(export_group)
        sl.addWidget(export_group)
        
        sl.addStretch()
        main_layout.addWidget(sidebar)

        # Center Splitter (Canvas + Right Layer Panel)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)

        # Right Sidebar (Layers)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setFixedWidth(300)
        right_layout.addWidget(QLabel("Layers:"))
        
        self.layer_list = QListWidget()
        self.layer_list.currentItemChanged.connect(self._on_layer_selected)
        self.layer_list.itemChanged.connect(self._on_layer_visibility_toggled)
        right_layout.addWidget(self.layer_list)

        # Layer management buttons
        lbtn_row = QHBoxLayout()
        up_btn = QPushButton("⬆"); up_btn.clicked.connect(lambda: self._move_layer(-1))
        down_btn = QPushButton("⬇"); down_btn.clicked.connect(lambda: self._move_layer(1))
        dup_btn = QPushButton("Dup"); dup_btn.clicked.connect(self._duplicate_layer)
        del_btn = QPushButton("Del"); del_btn.clicked.connect(self._delete_layer)
        lbtn_row.addWidget(up_btn); lbtn_row.addWidget(down_btn)
        lbtn_row.addWidget(dup_btn); lbtn_row.addWidget(del_btn)
        right_layout.addLayout(lbtn_row)

        right_layout.addWidget(QLabel("Layer Properties:"))
        right_layout.addWidget(self.layer_props)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Space"), self, self.toggle_play)
        QShortcut(QKeySequence("R"), self, self.canvas.reset_time)
        QShortcut(QKeySequence("C"), self, self.canvas.clear_all)
        QShortcut(QKeySequence("D"), self, lambda: self.draw_btn.toggle())
        QShortcut(QKeySequence("F"), self, self.fit_view)
        QShortcut(QKeySequence("Esc"), self, lambda: self.draw_btn.setChecked(False) if self.draw_btn.isChecked() else None)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_label = QLabel("Ready")
        sb.addWidget(self.status_label)
        self.fps_label = QLabel("FPS: 0")
        sb.addPermanentWidget(self.fps_label)

        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(250)

    def _refresh_status(self):
        self.fps_label.setText(f"FPS: {self.canvas.fps}")
        self.status_label.setText(
            f"Layers: {len(self.canvas.layers)}  |  Time: {self.canvas.time:.2f}  |  Zoom: {self.canvas.zoom:.2f}x"
        )

    def _populate_presets(self):
        for p in PRESETS:
            self.preset_combo.addItem(p["name"])

    # --- Layer Management ---

    def _refresh_layer_list(self):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for i, layer in enumerate(self.canvas.layers):
            item = QListWidgetItem(f"{i+1}. {layer.name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if layer.visible else Qt.Unchecked)
            item.setData(Qt.UserRole, i)
            self.layer_list.addItem(item)
        self.layer_list.blockSignals(False)

    def _on_layer_selected(self, current, previous):
        if not current:
            self.layer_props.load_layer(None)
            return
        idx = current.data(Qt.UserRole)
        if 0 <= idx < len(self.canvas.layers):
            self.layer_props.load_layer(self.canvas.layers[idx])

    def _on_layer_visibility_toggled(self, item):
        idx = item.data(Qt.UserRole)
        if 0 <= idx < len(self.canvas.layers):
            self.canvas.layers[idx].visible = (item.checkState() == Qt.Checked)
            self.canvas.update()

    def _move_layer(self, dir):
        row = self.layer_list.currentRow()
        if row < 0: return
        new_row = row + dir
        if 0 <= new_row < len(self.canvas.layers):
            layers = self.canvas.layers
            layers[row], layers[new_row] = layers[new_row], layers[row]
            self._refresh_layer_list()
            self.layer_list.setCurrentRow(new_row)
            self.canvas.update()

    def _delete_layer(self):
        row = self.layer_list.currentRow()
        if row < 0: return
        del self.canvas.layers[row]
        self._refresh_layer_list()
        if self.layer_list.count() > 0:
            self.layer_list.setCurrentRow(max(0, row-1))
        else:
            self.layer_props.load_layer(None)
        self.canvas.update()

    def _duplicate_layer(self):
        row = self.layer_list.currentRow()
        if row < 0: return
        orig = self.canvas.layers[row]
        new_layer = copy.deepcopy(orig)
        new_layer.name = orig.name + " Copy"
        new_layer.trail = []
        self.canvas.layers.insert(row+1, new_layer)
        self._refresh_layer_list()
        self.layer_list.setCurrentRow(row+1)
        self.canvas.update()

    # --- Actions ---

    def toggle_drawing_mode(self, checked):
        self.canvas.drawing_mode = checked
        self.canvas.playing = not checked
        self.play_btn.setText("▶ Play" if not self.canvas.playing else "⏸ Pause")
        self.draw_btn.setText("Exit Drawing Mode" if checked else "Enter Drawing Mode")
        self.canvas.update()

    def toggle_play(self):
        self.canvas.playing = not self.canvas.playing
        self.play_btn.setText("▶ Play" if not self.canvas.playing else "⏸ Pause")

    def _on_time_scrub(self, val):
        self.canvas.time = (val / 1000.0) * 2.0 * math.pi
        for layer in self.canvas.layers: layer.trail.clear()
        self.canvas.update()

    def fit_view(self):
        self.canvas.zoom = 1.0
        self.canvas.pan_x = 0.0
        self.canvas.pan_y = 0.0
        self.canvas.update()

    def pick_bg_color(self):
        c = QColorDialog.getColor(self.canvas.background_color, self)
        if c.isValid():
            self.canvas.background_color = c
            self.canvas.update()

    def import_svg(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open SVG", "", "SVG Files (*.svg)")
        if not fname: return
        try:
            strokes = parse_svg_file(fname)
            strokes = [s for s in strokes if len(s) > 5]
            for i, s in enumerate(strokes):
                self.canvas.add_layer_from_points(s, name=f"SVG Path {i+1}")
            self._refresh_layer_list()
            if strokes: self.draw_btn.setChecked(False)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to parse SVG:\n{e}")

    def load_preset(self):
        idx = self.preset_combo.currentIndex()
        if idx < 0: return
        preset = PRESETS[idx]
        t_vals = np.linspace(0, 2 * math.pi, 500, endpoint=False)
        points = []
        for t in t_vals:
            x, y = preset['fn'](t)
            points.append((x * 200, y * 200))
        self.canvas.add_layer_from_points(points, name=preset['name'])
        self._refresh_layer_list()
        self.draw_btn.setChecked(False)

    # --- Exporting ---

    def export_png(self):
        if not self.canvas.layers:
            QMessageBox.information(self, "Export", "Nothing to export.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save PNG", "fourier.png", "PNG Files (*.png)")
        if not fname: return
        img = self.canvas.render_to_image(
            include_epicycles=self.export_epicycles_chk.isChecked(),
            transparent=self.export_transparent_chk.isChecked()
        )
        img.save(fname)
        self.status_label.setText(f"Saved: {fname}")

    def export_gif(self):
        if not self.canvas.layers: return
        fname, _ = QFileDialog.getSaveFileName(self, "Save GIF", "fourier.gif", "GIF Files (*.gif)")
        if fname: self._run_animation_export(fname, fmt='gif')

    def export_mp4(self):
        if not self.canvas.layers: return
        fname, _ = QFileDialog.getSaveFileName(self, "Save MP4", "fourier.mp4", "MP4 Files (*.mp4)")
        if fname: self._run_animation_export(fname, fmt='mp4')

    def _get_export_resolution(self):
        choice = self.res_combo.currentText()
        if choice == "720p": return 1280, 720
        if choice == "1080p": return 1920, 1080
        if choice == "Square 720": return 720, 720
        return self.canvas.width(), self.canvas.height()

    def _run_animation_export(self, filename, fmt='gif'):
        was_playing = self.canvas.playing
        self.canvas.playing = False

        width, height = self._get_export_resolution()
        total_frames = self.frames_spin.value()
        fps = self.fps_spin.value()
        include_epi = self.export_epicycles_chk.isChecked()

        progress = QProgressBar(self)
        progress.setRange(0, total_frames)
        progress.setWindowTitle("Rendering...")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.show()

        frames = []
        original_time = self.canvas.time
        original_trails = [list(l.trail) for l in self.canvas.layers]
        for l in self.canvas.layers: l.trail = []

        try:
            for i in range(total_frames):
                progress.setValue(i)
                QApplication.processEvents()
                if progress.wasCanceled: break

                t = (i / total_frames) * 2.0 * math.pi
                self.canvas.time = t

                for layer in self.canvas.layers:
                    if len(layer.freqs) == 0: continue
                    x, y = render_frame_tip(
                        layer.freqs, layer.amps, layer.phases,
                        t, layer.max_terms, layer.centroid[0], -layer.centroid[1]
                    )
                    layer.trail.append((x, y))
                    if len(layer.trail) > layer.trail_length: layer.trail.pop(0)

                img = self.canvas.render_to_image(width, height, include_epicycles=include_epi)
                ptr = img.constBits()
                ptr.setsize(img.sizeInBytes())
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
                arr_bgr = arr[:, :, :3][:, :, ::-1].copy()
                frames.append(arr_bgr)
        finally:
            self.canvas.time = original_time
            for l, tr in zip(self.canvas.layers, original_trails): l.trail = tr
            self.canvas.playing = was_playing
            progress.close()

        try:
            if fmt == 'gif':
                frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
                imageio.mimsave(filename, frames_rgb, fps=fps, loop=0)
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                for f in frames: out.write(f)
                out.release()
            self.status_label.setText(f"Exported: {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to write file:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())