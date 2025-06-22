import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class NailDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, num_points=5):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_points = num_points
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
        logging.info(f"Found {len(self.images)} images in {img_dir}")

        self.valid_pairs = []
        for img_name in self.images:
            mask_name = img_name.rsplit('.', 1)[0] + '_mask.png'
            mask_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.valid_pairs.append((img_name, mask_name))
            else:
                logging.warning(f"Mask not found for {img_name}: {mask_path}")

        logging.info(f"Found {len(self.valid_pairs)} valid image-mask pairs")
        if len(self.valid_pairs) == 0:
            raise ValueError("No valid image-mask pairs found")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        logging.debug(f"Loading image: {img_path}")
        logging.debug(f"Loading mask: {mask_path}")

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0
        except Exception as e:
            logging.error(f"Error loading {img_name}: {e}")
            raise

        if np.any(mask):
            coords = np.where(mask > 0)
            num_available = len(coords[0])
            if num_available >= self.num_points:
                indices = np.random.choice(num_available, self.num_points, replace=False)
                prompt = [[int(coords[1][i]), int(coords[0][i])] for i in indices]
            else:
                prompt = [[int(coords[1][i]), int(coords[0][i])] for i in range(num_available)]
                prompt += [[random.randint(0, image.shape[1]-1), random.randint(0, image.shape[0]-1)] for _ in range(self.num_points - num_available)]
        else:
            prompt = [[random.randint(64, 192), random.randint(64, 192)] for _ in range(self.num_points)]

        prompt = np.array(prompt, dtype=np.float32)

        if self.transform:
            try:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
                scale_y = 1024 / image.shape[0]
                scale_x = 1024 / image.shape[1]
                prompt = prompt * np.array([[scale_x, scale_y]])
                prompt = prompt.astype(np.float32)
            except Exception as e:
                logging.error(f"Error augmenting {img_name}: {e}")
                raise

        mask = mask.unsqueeze(0) if mask.ndim == 2 else mask
        return image, mask, prompt
