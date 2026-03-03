import os
import urllib.request
import re
import json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

out_dir = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/assets/backgrounds/dynamic_pool"
os.makedirs(out_dir, exist_ok=True)

queries = [
    "wood texture high resolution",
    "marble texture seamless",
    "denim fabric texture",
    "rough concrete texture",
    "white cardboard paper texture",
    "dark wood grain texture",
    "granite stone texture",
    "leather texture high res",
    "sand texture seamless",
    "asphalt road texture",
]

base_url = "https://unsplash.com/napi/search/photos?query={}&per_page=3"

success = 0
for q in queries:
    req_url = base_url.format(urllib.parse.quote(q))
    req = urllib.request.Request(req_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            for item in data.get('results', []):
                img_url = item['urls']['regular']
                img_id = item['id']
                
                # download it
                slug = q.replace(" ", "_").replace("texture", "").replace("seamless", "").replace("high_resolution", "").replace("high_res", "")
                slug = "_".join(filter(None, slug.split("_")))
                
                path = os.path.join(out_dir, f"{slug}_{img_id}.jpg")
                if not os.path.exists(path):
                    dl_req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(dl_req) as dl_resp:
                        with open(path, 'wb') as f:
                            f.write(dl_resp.read())
                    print(f"Downloaded: {slug}_{img_id}.jpg")
                    success += 1
                if success >= 20:
                    break
    except Exception as e:
        print(f"Error on {q}: {e}")
        
    if success >= 20:
        break
    import time
    time.sleep(1)

print(f"Total downloaded: {success}")
