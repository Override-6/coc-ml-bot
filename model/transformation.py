import os
import random
import torch

import torchvision.transforms as T
from PIL import ImageDraw, Image

IMAGE_SIZE = 2000

class AddRandomWhiteDots:
    def __init__(self, num_circles, min_radius, max_radius):
        self.num_circles = num_circles
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, img):
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size

        for _ in range(self.num_circles):
            radius = random.randint(self.min_radius, self.max_radius)
            x = random.randint(radius, width - radius)
            y = random.randint(radius, height - radius)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='white')

        img.save("/tmp/created_image.png")

        return img


class ToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, tensor):
        return tensor.to(self.device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

transforms = T.Compose([
    AddRandomWhiteDots(num_circles=450, min_radius=2, max_radius=5),
    T.ToTensor(),
    ToDevice(device),
    T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.2),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=4),
    T.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.0)),
])


# Apply the transformations to each image in the dataset before training
def apply_transforms_to_folder(images_folder, tranformations_per_images, transforms):
    # Load your dataset here and apply the transformations
    from pathlib import Path

    folder_path = Path(images_folder)

    for img_path in os.listdir(images_folder):  # Adjust the pattern to match your dataset
        img = Image.open(img_path)

        print(f"transforming image {img_path}\r")

        for x in range(tranformations_per_images):
            transformed_img = transforms(img)
            output_img_path = folder_path / img_path.name.replace('.png', f'-transform-{x}.png')
            transformed_img_pil = T.ToPILImage()(transformed_img)
            transformed_img_pil.save(output_img_path)

apply_transforms_to_folder("../../coc-base-generator/out/images/train", 3, transforms)
apply_transforms_to_folder("../../coc-base-generator/out/images/val", 3, transforms)
