import re
import numpy as np

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