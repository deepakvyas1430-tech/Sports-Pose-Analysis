import math
import cv2
import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points a, b, c (where b is the vertex).
    Points should be (x, y) coordinates or [x, y].
    Returns angle in degrees.
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def calculate_vertical_angle(a, b):
    """
    Calculates the angle of the line segment ab relative to the vertical axis.
    a: top point (e.g., shoulder)
    b: bottom point (e.g., hip)
    Returns angle in degrees (0 = vertical).
    """
    a = np.array(a)
    b = np.array(b)
    
    # Vector ab
    vector = b - a
    
    # Vertical vector (pointing down)
    vertical = np.array([0, 1])
    
    # Dot product
    dot_product = np.dot(vector, vertical)
    magnitude = np.linalg.norm(vector)
    
    if magnitude == 0:
        return 0
        
    # Cosine of angle
    cos_angle = dot_product / magnitude
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180.0 / np.pi
    
    return angle

def draw_text(img, text, position, color=(255, 255, 255), scale=0.6, thickness=2):
    """Helper to draw text with a black outline for better visibility."""
    x, y = position
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
