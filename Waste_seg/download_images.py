import pandas as pd
import requests
import os
import json

# ---------------------------
# ⚙️ CONFIGURATION
# ---------------------------
CSV_PATH = "all_image_urls.csv"           # your CSV file (with jpg_url, png_url)
JSON_PATH = "annotations.json"            # official TACO JSON annotation file
OUTPUT_IMG_DIR = "TACO/images"            # where to save images
OUTPUT_JSON_PATH = "TACO/filtered_annotations.json"  # filtered annotations output
LOG_PATH = "downloaded_image_ids.txt"     # text file to store downloaded image IDs
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# ---------------------------
# 📂 LOAD FILES
# ---------------------------
print("📖 Loading CSV and JSON...")
df = pd.read_csv(CSV_PATH, header=None, names=['jpg_url', 'png_url'])

with open(JSON_PATH, "r") as f:
    data = json.load(f)

images_info = data["images"]
annotations_info = data["annotations"]

# ---------------------------
# 🔗 BUILD IMAGE DICTIONARY
# ---------------------------
image_dict = {}
for item in images_info:
    image_id = str(item["id"])
    # Prefer flickr_640_url if available, else fallback to flickr_url
    url = item.get("flickr_640_url") or item.get("flickr_url")
    image_dict[image_id] = url

print(f"🔍 Found {len(image_dict)} image entries in JSON")

# ---------------------------
# 💾 DOWNLOAD FUNCTION
# ---------------------------
def download_file(url, save_path):
    try:
        if not isinstance(url, str) or not url.startswith("http"):
            print(f"❌ Invalid URL skipped: {url}")
            return False
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
        else:
            print(f"⚠️ Failed ({r.status_code}): {url}")
            return False
    except Exception as e:
        print(f"⚠️ Error downloading {url}: {e}")
        return False

# ---------------------------
# ⬇️ DOWNLOAD LOOP
# ---------------------------
success = []
failed = []

print("\n🚀 Starting downloads...")

for i, row in df.iterrows():
    image_id = str(i + 1)  # assuming CSV rows align with JSON IDs
    url = image_dict.get(image_id, row['jpg_url'])
    image_name = f"{image_id.zfill(5)}.jpg"
    save_path = os.path.join(OUTPUT_IMG_DIR, image_name)

    if download_file(url, save_path):
        success.append(int(image_id))
    else:
        failed.append(int(image_id))

    if (i + 1) % 50 == 0:
        print(f"📦 Downloaded {i + 1}/{len(df)} images...")

# ---------------------------
# 🧾 SAVE SUCCESSFUL IDS
# ---------------------------
with open(LOG_PATH, "w") as f:
    f.write("\n".join(map(str, success)))

print(f"\n✅ Download complete: {len(success)} successful, {len(failed)} failed.")
print(f"🗂️ Saved downloaded image IDs to '{LOG_PATH}'")

# ---------------------------
# 🧹 FILTER ANNOTATIONS
# ---------------------------
print("\n🧩 Filtering annotations for downloaded images...")

valid_ids = set(success)

filtered_images = [img for img in images_info if img["id"] in valid_ids]
filtered_annotations = [a for a in annotations_info if a["image_id"] in valid_ids]

filtered_data = {
    "info": data["info"],
    "licenses": data.get("licenses", []),
    "categories": data["categories"],
    "images": filtered_images,
    "annotations": filtered_annotations
}

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(filtered_data, f, indent=4)

print(f"✅ Saved filtered annotations to '{OUTPUT_JSON_PATH}'")
print(f"📊 Final dataset: {len(filtered_images)} images, {len(filtered_annotations)} annotations.")

