import os
import urllib.request
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context
out_dir = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/assets/backgrounds/dynamic_pool"
os.makedirs(out_dir, exist_ok=True)

# Unsplash image IDs that are known to hold good textures
unsplash_ids = {
    "wood_light.jpg": "1550684848-fac1c5b4e853",
    "wood_dark.jpg": "1546484475-7f7cd557a55c",
    "marble_white.jpg": "1551024506-0cb98b64b197",
    "fabric_linen.jpg": "1605221950074-b5a9cd2ec06f",
    "leather_brown.jpg": "1563823293836-728b975e50db",
    "granite.jpg": "1518057111178-44a106bad636",
    "paper_crumpled.jpg": "1603503350109-1a067341e9c2",
    "fabric_knit.jpg": "1588698188174-89311e97d4dd",
    "plastic_blue.jpg": "1558591710-4b4a1ae0f04d",
    "rust_metal.jpg": "1497933923425-4c0383cd89e1",
    "sand.jpg": "1506501254395-654b9d0dc079",
    "asphalt.jpg": "1518174092523-9c86460e5758",
    "cardboard.jpg": "1564507004663-b6dfb3c824d5",
    "wood_planks.jpg": "1510617303351-40be3f7e1553",
    "concrete_rough.jpg": "1517482322303-3ea76b8bd7c5"
}

success = 0
for name, uid in unsplash_ids.items():
    path = os.path.join(out_dir, name)
    url = f"https://images.unsplash.com/photo-{uid}?w=1000&q=80"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(path, 'wb') as f:
                f.write(response.read())
        print(f"Downloaded: {name}")
        success += 1
    except Exception as e:
        print(f"Failed {name}: {e}")
    time.sleep(1)

print(f"Total downloaded: {success}/15")
