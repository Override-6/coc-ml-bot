import os

import requests
from PIL import Image
from bs4 import BeautifulSoup

buildings = ["Cannon", "Archer Tower", "Mortar", "Air Defense", "Wizard Tower", "Air Sweeper", "Hidden Tesla",
             "Bomb Tower", "X-Bow", "Inferno Tower", "Eagle Artillery", "Scattershot", "Builder's Hut", "Spell Tower",
             "Monolith", "Ricochet Cannon", "Multi-Archer Tower", "Giga Tesla", "Bomb",
             "Spring Trap", "Air Bomb", "Giant Bomb", "Seeking Air Mine", "Skeleton Trap", "Tornado Trap",
             "Town Hall", "Gold Mine", "Elixir Collector", "Dark Elixir Drill", "Gold Storage", "Elixir Storage",
             "Dark Elixir Storage", "Clan Castle (Treasury)",
             "Army Camp", "Barracks", "Dark Barracks", "Laboratory", "Spell Factory", "Dark Spell Factory",
             "Blacksmith", "Workshop", "Barbarian King Altar", "Archer Queen Altar", "Grand Warden Altar",
             "Royal Champion Altar", "Pet House", "Boat", "Airship", "Forge", "Decorations", "Obstacles", "Loot Cart",
             "Strongman's Caravan", "Super Sauna", "Builder's Hut",
             ]

dataset_path = 'datasets/buildings'
images_path = f"{dataset_path}/images"
labels_path = f"{dataset_path}/labels"
os.makedirs(os.path.join(images_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(images_path, 'val'), exist_ok=True)
os.makedirs(os.path.join(labels_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_path, 'val'), exist_ok=True)

class_id = 0
class_names = []

for building in buildings:
    building = building.replace(" ", "_")
    print(f"Scrapping building {building}")
    rq = requests.get(f"https://clashofclans.fandom.com/wiki/{building}")

    if rq.status_code == 404:
        rq = requests.get(f"https://clashofclans.fandom.com/wiki/{building}/Home_Village")

    html = rq.content.decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.select(".flexbox-display.bold-text img"):
        url = img.get("data-src")
        if url is None:
            continue

        class_name = url.split("/")[-5][:-4]

        image_path = f"{dataset_path}/images/train/{class_name}.png"
        image_path_val = f"{dataset_path}/images/val/{class_name}.png"
        label_path = f"{dataset_path}/labels/train/{class_name}.txt"
        label_path_val = f"{dataset_path}/labels/val/{class_name}.txt"

        class_names.append(class_name)

        if os.path.exists(image_path):
            class_id += 1
            continue

        img = Image.open(requests.get(url, stream=True).raw)
        img = img.crop(img.getbbox())
        img.save(image_path, "PNG")

        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        img.save(image_path_val, "PNG")

        with open(label_path, 'w') as label_file:
            label_file.write(f'{class_id} 0.5 0.5 1.0 1.0\n')
        with open(label_path_val, 'w') as label_file:
            label_file.write(f'{class_id} 0.5 0.5 1.0 1.0\n')

        class_id += 1

print("Generating YOLO model configuration")

content = f"""
train: ./images/train
val: ./images/val

nc: {len(class_names)}
names: {class_names}
"""

with open("datasets/buildings/data.yaml", 'w') as file:
    file.write(content)
