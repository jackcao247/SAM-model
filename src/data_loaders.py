import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import NailDataset
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_transforms():
    return A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})

def collate_fn(batch):
    images, masks, prompts = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    prompts = torch.stack([torch.tensor(p, dtype=torch.float32) for p in prompts], dim=0)
    return images, masks, prompts

def get_loaders():
    try:
        train_dataset = NailDataset(
            img_dir='/content/train/images',
            mask_dir='/content/train/masks',
            transform=get_transforms(),
            num_points=5
        )
        valid_dataset = NailDataset(
            img_dir='/content/valid/images',
            mask_dir='/content/valid/masks',
            transform=get_transforms(),
            num_points=5
        )
        test_dataset = NailDataset(
            img_dir='/content/test/images',
            mask_dir='/content/test/masks',
            transform=get_transforms(),
            num_points=5
        )
        logging.info(f"Train dataset: {len(train_dataset)} valid pairs")
        logging.info(f"Valid dataset: {len(valid_dataset)} valid pairs")
        logging.info(f"Test dataset: {len(test_dataset)} valid pairs")
    except Exception as e:
        logging.error(f"Error initializing datasets: {e}")
        raise

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    return train_loader, valid_loader, test_loader