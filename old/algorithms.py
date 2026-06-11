import math
import numpy as np
from numba import jit

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