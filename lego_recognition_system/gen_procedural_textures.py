import os
import numpy as np
from PIL import Image
import random
import colorsys

out_dir = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/assets/backgrounds/dynamic_pool"
os.makedirs(out_dir, exist_ok=True)

size = (1000, 1000)

def generate_noise(w, h, color_base):
    # Gaussian noise around a base color
    noise = np.random.normal(0, 30, (h, w, 3))
    base_img = np.full((h, w, 3), color_base, dtype=np.float32)
    img = np.clip(base_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def generate_wood_grain(w, h, color1, color2):
    # Simulated wood grain via sine wave + noise
    img = np.zeros((h, w, 3), dtype=np.float32)
    X, Y = np.meshgrid(np.linspace(0, 10, w), np.linspace(0, 10, h))
    turb = np.random.normal(0, 0.5, (h, w))
    # low pass filter the noise
    from scipy.ndimage import gaussian_filter
    turb = gaussian_filter(turb, sigma=5)
    
    pattern = np.sin(Y * 2 + turb * 10) * 0.5 + 0.5
    for c in range(3):
        img[:,:,c] = color1[c] * pattern + color2[c] * (1 - pattern)
    
    img = np.clip(img + np.random.normal(0, 10, (h, w, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def generate_gradient(w, h, color1, color2):
    X = np.linspace(0, 1, w)
    img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        row = color1[c] * (1 - X) + color2[c] * X
        img[:,:,c] = np.tile(row, (h, 1))
    
    img = np.clip(img + np.random.normal(0, 15, (h, w, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def rand_color():
    h = random.random()
    s = random.uniform(0.2, 0.8)
    v = random.uniform(0.3, 0.9)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r*255), int(g*255), int(b*255))

for i in range(20):
    choice = random.choice(['noise', 'wood', 'gradient'])
    try:
        if choice == 'noise':
            img = generate_noise(*size, rand_color())
            name = f"proc_noise_{i}.jpg"
        elif choice == 'wood':
            c1 = rand_color()
            c2 = (max(0, c1[0]-40), max(0, c1[1]-40), max(0, c1[2]-40))
            img = generate_wood_grain(*size, c1, c2)
            name = f"proc_wood_{i}.jpg"
        else:
            img = generate_gradient(*size, rand_color(), rand_color())
            name = f"proc_gradient_{i}.jpg"
            
        img.save(os.path.join(out_dir, name), quality=85)
        print(f"Generated {name}")
    except Exception as e:
        print(f"Failed {i}: {e}")

print("Done procedural generation.")
