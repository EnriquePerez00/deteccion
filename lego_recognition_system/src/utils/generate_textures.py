import os
import random
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

# Configuración básica
TEXTURES_DIR = Path(__file__).parent.parent / "assets" / "backgrounds" / "dynamic_pool"
TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
SIZE = 1024
COUNT_PER_TYPE = 10

def _normalize_img(arr):
    arr = arr - np.min(arr)
    arr = arr / np.max(arr)
    return (arr * 255).astype(np.uint8)

def generate_noise(filename):
    """Ruido granular (estilo TV) con algo de desenfoque y variación de color."""
    base_color = np.random.randint(50, 200, size=3)
    noise = np.random.randint(-50, 50, size=(SIZE, SIZE, 3))
    arr = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    # Desenfoque leve aleatorio
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    img.save(filename)

def generate_wood(filename):
    """Madera sintética usando ondas sinusoidales perturbadas con ruido."""
    x = np.linspace(0, 10, SIZE)
    y = np.linspace(0, 10, SIZE)
    X, Y = np.meshgrid(x, y)
    
    # Ruido de baja frecuencia
    noise = np.random.randn(SIZE, SIZE)
    import scipy.ndimage
    noise = scipy.ndimage.gaussian_filter(noise, sigma=15)
    noise = _normalize_img(noise) / 255.0
    
    # Patrón de vetas
    frequency = random.uniform(5, 15)
    turbulence = random.uniform(2, 5)
    
    pattern = np.sin((X + noise * turbulence) * frequency)
    pattern = (pattern + 1) / 2.0  # Normalizar 0 a 1
    
    # Mapeo a colores de madera
    color_dark = np.array([random.randint(60, 100), random.randint(30, 50), random.randint(10, 30)])
    color_light = np.array([random.randint(150, 200), random.randint(100, 150), random.randint(50, 100)])
    
    wood = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for i in range(3):
        wood[:,:,i] = (pattern * color_light[i] + (1 - pattern) * color_dark[i]).astype(np.uint8)
        
    img = Image.fromarray(wood)
    # Añadir un poco de grano
    grano = np.random.randint(-10, 10, size=(SIZE, SIZE, 3))
    wood_gran = np.clip(wood.astype(int) + grano, 0, 255).astype(np.uint8)
    img = Image.fromarray(wood_gran)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img.save(filename)

def generate_marble(filename):
    """Mármol mediante ruido Perlin simulado con filtros gaussianos multi-escala."""
    noise_base = np.random.randn(SIZE, SIZE)
    import scipy.ndimage
    
    # Suma de octavas de ruido
    noise = np.zeros((SIZE, SIZE))
    amplitudes = [1.0, 0.5, 0.25, 0.125]
    sigmas = [64, 32, 16, 8]
    
    for a, s in zip(amplitudes, sigmas):
        n = scipy.ndimage.gaussian_filter(np.random.randn(SIZE, SIZE), sigma=s)
        noise += a * n
        
    # Crear bandas (turbulencia)
    turbulence = random.uniform(3, 8)
    x = np.linspace(0, 10, SIZE)
    X, _ = np.meshgrid(x, x)
    
    pattern = np.sin(X * turbulence + noise * 10)
    pattern = np.abs(pattern) # Vetas definidas
    pattern = 1.0 - pattern # Invertir para vetas oscuras
    
    # Mapeo de color
    base_color = np.array([random.randint(200, 250)] * 3)
    vein_color = np.array([random.randint(50, 100)] * 3)
    
    marble = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for i in range(3):
        marble[:,:,i] = (pattern * base_color[i] + (1 - pattern) * vein_color[i]).astype(np.uint8)
        
    img = Image.fromarray(marble)
    img.save(filename)

def generate_metal(filename):
    """Metal cepillado mediante ruido estirado direccionalmente."""
    noise = np.random.randint(0, 255, size=(SIZE, SIZE))
    import scipy.ndimage
    # Estirar horizontalmente o verticalmente
    if random.choice([True, False]):
        metal = scipy.ndimage.gaussian_filter(noise, sigma=(1, 50))
    else:
        metal = scipy.ndimage.gaussian_filter(noise, sigma=(50, 1))
        
    metal = _normalize_img(metal)
    
    # Tinte oscuro
    base = random.randint(50, 150)
    metal_rgb = np.stack([metal]*3, axis=-1)
    
    # Aplicar tinte ligero
    tint = np.array([random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)])
    metal_rgb = np.clip(metal_rgb * tint * (base/128.0), 0, 255).astype(np.uint8)
    
    img = Image.fromarray(metal_rgb)
    img.save(filename)

def generate_plastic(filename):
    """Plástico liso con ligera variación de iluminación/baja frecuencia."""
    color = np.array([random.randint(20, 230), random.randint(20, 230), random.randint(20, 230)])
    
    # Gradiente suave o viñeta
    x = np.linspace(-1, 1, SIZE)
    y = np.linspace(-1, 1, SIZE)
    X, Y = np.meshgrid(x, y)
    
    # Gradiente radial (iluminación tipo spot)
    cx, cy = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    gradient = 1.0 - (dist / dist.max()) * random.uniform(0.2, 0.6)
    
    plastic = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for i in range(3):
        plastic[:,:,i] = np.clip(color[i] * gradient, 0, 255).astype(np.uint8)
        
    img = Image.fromarray(plastic)
    # Añadir micromanchitas (ruido sutilísimo)
    noise = np.random.randint(-3, 3, size=(SIZE, SIZE, 3))
    plastic_noisy = np.clip(plastic.astype(int) + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(plastic_noisy)
    img.save(filename)


if __name__ == "__main__":
    print(f"Generando {COUNT_PER_TYPE * 5} texturas procedurales en: {TEXTURES_DIR}")
    
    # Limpiar directorio previo si existe pero conservando el folder
    for f in os.listdir(TEXTURES_DIR):
        if f.endswith(".jpg"):
            os.remove(TEXTURES_DIR / f)
            
    # Generar iterativamente
    for i in range(COUNT_PER_TYPE):
        generate_noise(TEXTURES_DIR / f"noise_{i:02d}.jpg")
        generate_wood(TEXTURES_DIR / f"wood_{i:02d}.jpg")
        generate_marble(TEXTURES_DIR / f"marble_{i:02d}.jpg")
        generate_metal(TEXTURES_DIR / f"metal_{i:02d}.jpg")
        generate_plastic(TEXTURES_DIR / f"plastic_{i:02d}.jpg")
        
    print("¡Generación de fondos completada con éxito!")
