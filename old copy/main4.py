import numpy as np
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Line, Color, Ellipse, InstructionGroup
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from PIL import Image, ImageDraw
import cv2
import io

class FourierDrawingWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drawing = []
        self.fourier_coeffs = []
        self.time = 0
        self.path = []
        self.speed = 1.0
        self.animation_event = None
        self.is_animating = False
        self.show_vectors = True
        self.show_circles = True
        self.max_vectors = 50
        self.bind(size=self.update_graphics, pos=self.update_graphics)
        
        # For storing animation frames
        self.animation_frames = []
        self.recording = False

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if not self.is_animating:
                self.drawing = [touch.pos]
                self.path = []
                self.time = 0
                self.canvas.clear()
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos) and len(self.drawing) > 0 and not self.is_animating:
            self.drawing.append(touch.pos)
            with self.canvas:
                Color(0.8, 0.8, 1, 1)  # Light blue
                Line(points=[self.drawing[-2][0], self.drawing[-2][1], 
                             self.drawing[-1][0], self.drawing[-1][1]], 
                     width=2, cap='round', joint='round')
            return True
        return super().on_touch_move(touch)

    def compute_fourier(self):
        if len(self.drawing) < 3:
            return False
            
        # Convert drawing to complex numbers
        points = [complex(p[0], p[1]) for p in self.drawing]
        
        # Compute DFT using FFT for better performance
        coeffs = np.fft.fftshift(np.fft.fft(points)) / len(points)
        freqs = np.fft.fftshift(np.fft.fftfreq(len(points)))
        
        # Create list of (amplitude, phase, frequency) tuples
        self.fourier_coeffs = []
        for i in range(len(coeffs)):
            if abs(coeffs[i]) > 1e-10:  # Filter out very small coefficients
                self.fourier_coeffs.append((
                    abs(coeffs[i]),
                    np.angle(coeffs[i]),
                    freqs[i]
                ))
        
        # Sort by amplitude (descending) and limit to max_vectors
        self.fourier_coeffs.sort(key=lambda x: x[0], reverse=True)
        self.fourier_coeffs = self.fourier_coeffs[:self.max_vectors]
        
        return True

    def start_animation(self):
        if not self.fourier_coeffs:
            return False
            
        self.path = []
        self.time = 0
        self.is_animating = True
        if self.animation_event:
            Clock.unschedule(self.animation_event)
        self.animation_event = Clock.schedule_interval(self.update_animation, 1.0/60.0)
        return True

    def stop_animation(self):
        if self.animation_event:
            Clock.unschedule(self.animation_event)
            self.animation_event = None
        self.is_animating = False

    def update_animation(self, dt):
        self.time += 0.01 * self.speed
        if self.time > 2 * np.pi:  # Complete one full cycle
            self.time = 0
            self.path = []
        self.update_graphics()

    def update_graphics(self, *args):
        # Clear only the animation part, keep the drawing
        self.canvas.clear()
        
        # Redraw user drawing
        if self.drawing and not self.is_animating:
            with self.canvas:
                Color(0.8, 0.8, 1, 1)  # Light blue
                points = []
                for p in self.drawing:
                    points.extend(p)
                Line(points=points, width=2, cap='round', joint='round')
        
        # Draw Fourier animation
        if self.fourier_coeffs and self.is_animating:
            x, y = self.fourier_animation(self.time)
            self.path.append((x, y))
            
            # Draw the reconstructed path
            if len(self.path) > 1:
                with self.canvas:
                    Color(0, 1, 0.5, 1)  # Bright green
                    points = []
                    for p in self.path:
                        points.extend(p)
                    Line(points=points, width=2, cap='round', joint='round')
                    
                    # Draw endpoint
                    Color(1, 0, 0, 1)  # Red endpoint
                    Ellipse(pos=(x-3, y-3), size=(6, 6))

    def fourier_animation(self, t):
        if not self.fourier_coeffs:
            return self.center_x, self.center_y

        # Start from center
        x = self.center_x
        y = self.center_y
        
        # Scale factor based on widget size
        scale = min(self.width, self.height) * 0.3

        # Draw vectors and circles
        if self.show_vectors or self.show_circles:
            with self.canvas:
                for i, (amp, phase, freq) in enumerate(self.fourier_coeffs):
                    prev_x, prev_y = x, y
                    
                    # Calculate vector endpoint
                    dx = scale * amp * np.cos(freq * t + phase)
                    dy = scale * amp * np.sin(freq * t + phase)
                    x += dx
                    y += dy

                    # Draw circles showing rotation
                    if self.show_circles and i < 10:  # Limit circles for performance
                        radius = scale * amp
                        if radius > 1:  # Only draw visible circles
                            Color(0.5, 0.5, 0.8, 0.3)  # Light blue, semi-transparent
                            Line(circle=(prev_x, prev_y, radius), width=1)
                    
                    # Draw vectors
                    if self.show_vectors:
                        # Color based on frequency (low freq = warm, high freq = cool)
                        norm_freq = abs(freq) / (max([abs(f[2]) for f in self.fourier_coeffs]) + 1e-10)
                        if i == 0:  # DC component
                            Color(1, 1, 0, 1)  # Yellow for DC
                        elif freq > 0:
                            Color(1, 0.5, 0, 0.8)  # Orange for positive frequencies
                        elif freq < 0:
                            Color(0, 0.7, 1, 0.8)  # Blue for negative frequencies
                        else:
                            Color(0.8, 0.2, 0.8, 0.8)  # Purple for zero frequency
                        
                        Line(points=[prev_x, prev_y, x, y], width=1.5, cap='round')

        return x, y

    def reset_drawing(self):
        self.drawing = []
        self.path = []
        self.fourier_coeffs = []
        self.time = 0
        self.is_animating = False
        self.stop_animation()
        self.canvas.clear()

    def change_speed(self, value):
        self.speed = value / 50.0

    def toggle_vectors(self):
        self.show_vectors = not self.show_vectors

    def toggle_circles(self):
        self.show_circles = not self.show_circles

    def set_max_vectors(self, value):
        self.max_vectors = int(value)
        if self.fourier_coeffs:
            # Re-sort and limit coefficients
            self.fourier_coeffs.sort(key=lambda x: x[0], reverse=True)
            self.fourier_coeffs = self.fourier_coeffs[:self.max_vectors]

    def export_gif(self):
        if not self.fourier_coeffs:
            return False
            
        frames = []
        N_frames = 300
        width, height = int(self.width), int(self.height)
        scale = min(width, height) * 0.3

        for i in range(N_frames):
            t = 2 * np.pi * i / N_frames
            img = Image.new("RGB", (width, height), (15, 15, 35))
            draw = ImageDraw.Draw(img)
            
            # Draw user drawing in background
            if self.drawing:
                draw.line(self.drawing, fill=(100, 150, 255, 180), width=2)
            
            # Draw Fourier animation
            x = width / 2
            y = height / 2
            points = [(x, y)]
            
            for amp, phase, freq in self.fourier_coeffs:
                prev_x, prev_y = x, y
                dx = scale * amp * np.cos(freq * t + phase)
                dy = scale * amp * np.sin(freq * t + phase)
                x += dx
                y += dy
                points.append((x, y))
            
            # Draw vectors
            for i in range(len(points) - 1):
                norm_freq = abs(self.fourier_coeffs[i][2]) / (max([abs(f[2]) for f in self.fourier_coeffs]) + 1e-10)
                if i == 0:
                    color = (255, 255, 0)  # Yellow for DC
                elif self.fourier_coeffs[i][2] > 0:
                    color = (255, 128, 0)  # Orange for positive
                else:
                    color = (0, 179, 255)  # Blue for negative
                draw.line([points[i], points[i+1]], fill=color, width=2)
            
            # Draw path
            if len(points) > 1:
                draw.line(points, fill=(0, 255, 128), width=2)
            
            frames.append(img)
            
        frames[0].save("fourier_animation.gif", save_all=True, append_images=frames[1:], 
                       optimize=True, duration=33, loop=0)
        return True

    def export_mp4(self):
        if not self.fourier_coeffs:
            return False
            
        width, height = int(self.width), int(self.height)
        scale = min(width, height) * 0.3
        N_frames = 300
        out = cv2.VideoWriter('fourier_animation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for i in range(N_frames):
            t = 2 * np.pi * i / N_frames
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:] = [15, 15, 35]  # Dark blue background
            
            # Draw user drawing in background
            if self.drawing:
                points = np.array(self.drawing, dtype=np.int32)
                cv2.polylines(img, [points], False, (100, 150, 255), 2)
            
            # Draw Fourier animation
            x = width / 2
            y = height / 2
            points = [(int(x), int(y))]
            
            for amp, phase, freq in self.fourier_coeffs:
                prev_x, prev_y = x, y
                dx = scale * amp * np.cos(freq * t + phase)
                dy = scale * amp * np.sin(freq * t + phase)
                x += dx
                y += dy
                points.append((int(x), int(y)))
            
            # Draw vectors
            for i in range(len(points) - 1):
                if i == 0:
                    color = (0, 255, 255)  # Yellow for DC
                elif self.fourier_coeffs[i][2] > 0:
                    color = (0, 128, 255)  # Orange for positive
                else:
                    color = (255, 179, 0)  # Blue for negative
                cv2.line(img, points[i], points[i+1], color, 2)
            
            # Draw path
            if len(points) > 1:
                cv2.polylines(img, [np.array(points, dtype=np.int32)], False, (128, 255, 0), 2)
            
            out.write(img)
            
        out.release()
        return True

    @property
    def center_x(self):
        return self.width / 2

    @property
    def center_y(self):
        return self.height / 2

class FourierDrawingApp(App):
    def build(self):
        self.title = "Fourier Series Visualizer"
        Window.size = (1200, 800)
        
        main_layout = BoxLayout(orientation='horizontal', padding=10, spacing=10)
        
        # Left panel for controls
        control_panel = BoxLayout(orientation='vertical', size_hint_x=0.3, spacing=10)
        
        # Info panel
        info_panel = BoxLayout(orientation='vertical', size_hint_y=None, height=100)
        self.info_label = Label(
            text="Draw a shape, then compute its Fourier transform",
            text_size=(300, None),
            halign='center',
            valign='middle'
        )
        info_panel.add_widget(self.info_label)
        control_panel.add_widget(info_panel)
        
        # Control buttons
        button_grid = GridLayout(cols=2, spacing=5, size_hint_y=None, height=200)
        
        self.compute_btn = Button(text="Compute Fourier")
        self.compute_btn.bind(on_press=self.compute_fourier)
        button_grid.add_widget(self.compute_btn)
        
        self.animate_btn = Button(text="Start Animation")
        self.animate_btn.bind(on_press=self.toggle_animation)
        button_grid.add_widget(self.animate_btn)
        
        self.reset_btn = Button(text="Reset")
        self.reset_btn.bind(on_press=self.reset_drawing)
        button_grid.add_widget(self.reset_btn)
        
        self.vector_toggle_btn = Button(text="Hide Vectors")
        self.vector_toggle_btn.bind(on_press=self.toggle_vectors)
        button_grid.add_widget(self.vector_toggle_btn)
        
        self.circle_toggle_btn = Button(text="Hide Circles")
        self.circle_toggle_btn.bind(on_press=self.toggle_circles)
        button_grid.add_widget(self.circle_toggle_btn)
        
        self.export_gif_btn = Button(text="Export GIF")
        self.export_gif_btn.bind(on_press=self.export_gif)
        button_grid.add_widget(self.export_gif_btn)
        
        self.export_mp4_btn = Button(text="Export MP4")
        self.export_mp4_btn.bind(on_press=self.export_mp4)
        button_grid.add_widget(self.export_mp4_btn)
        
        control_panel.add_widget(button_grid)
        
        # Speed control
        speed_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=80)
        speed_layout.add_widget(Label(text="Animation Speed:"))
        self.speed_slider = Slider(min=1, max=200, value=50)
        self.speed_slider.bind(value=self.change_speed)
        speed_layout.add_widget(self.speed_slider)
        control_panel.add_widget(speed_layout)
        
        # Vector count control
        vector_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=80)
        vector_layout.add_widget(Label(text="Max Vectors:"))
        self.vector_slider = Slider(min=5, max=200, value=50)
        self.vector_slider.bind(value=self.change_vectors)
        vector_layout.add_widget(self.vector_slider)
        control_panel.add_widget(vector_layout)
        
        # Technical info
        tech_info = BoxLayout(orientation='vertical')
        tech_info.add_widget(Label(text="Technical Details:", size_hint_y=None, height=30))
        self.tech_label = Label(
            text="No Fourier data computed",
            text_size=(300, None),
            halign='left',
            valign='top'
        )
        tech_info.add_widget(self.tech_label)
        control_panel.add_widget(tech_info)
        
        # Drawing widget
        self.drawing_widget = FourierDrawingWidget()
        
        main_layout.add_widget(control_panel)
        main_layout.add_widget(self.drawing_widget)
        
        return main_layout

    def compute_fourier(self, instance):
        if self.drawing_widget.compute_fourier():
            self.info_label.text = "Fourier transform computed successfully!"
            coeffs = self.drawing_widget.fourier_coeffs
            if coeffs:
                dc = coeffs[0][0] if coeffs else 0
                max_freq = max([abs(c[2]) for c in coeffs]) if coeffs else 0
                self.tech_label.text = (
                    f"Coefficients: {len(coeffs)}\n"
                    f"DC Component: {dc:.2f}\n"
                    f"Max Frequency: {max_freq:.2f}\n"
                    f"Fundamental: {abs(coeffs[1][2]) if len(coeffs) > 1 else 0:.2f}"
                )
        else:
            self.info_label.text = "Draw a more complex shape first!"

    def toggle_animation(self, instance):
        if instance.text == "Start Animation":
            if self.drawing_widget.start_animation():
                instance.text = "Stop Animation"
                self.compute_btn.disabled = True
            else:
                self.info_label.text = "Compute Fourier transform first!"
        else:
            self.drawing_widget.stop_animation()
            instance.text = "Start Animation"
            self.compute_btn.disabled = False

    def reset_drawing(self, instance):
        self.drawing_widget.reset_drawing()
        self.info_label.text = "Draw a shape, then compute its Fourier transform"
        self.tech_label.text = "No Fourier data computed"
        self.animate_btn.text = "Start Animation"
        self.compute_btn.disabled = False

    def change_speed(self, instance, value):
        self.drawing_widget.change_speed(value)

    def change_vectors(self, instance, value):
        self.drawing_widget.set_max_vectors(value)

    def toggle_vectors(self, instance):
        self.drawing_widget.toggle_vectors()
        instance.text = "Hide Vectors" if self.drawing_widget.show_vectors else "Show Vectors"

    def toggle_circles(self, instance):
        self.drawing_widget.toggle_circles()
        instance.text = "Hide Circles" if self.drawing_widget.show_circles else "Show Circles"

    def export_gif(self, instance):
        if self.drawing_widget.export_gif():
            self.info_label.text = "GIF saved: fourier_animation.gif"
        else:
            self.info_label.text = "Compute Fourier transform first!"

    def export_mp4(self, instance):
        if self.drawing_widget.export_mp4():
            self.info_label.text = "MP4 saved: fourier_animation.mp4"
        else:
            self.info_label.text = "Compute Fourier transform first!"

if __name__ == '__main__':
    FourierDrawingApp().run()