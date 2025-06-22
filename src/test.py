import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import os
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sam_test_local.log', mode='a'),
        logging.StreamHandler()
    ]
)

def get_transforms():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def compute_metrics(pred_mask, gt_mask):
    """Compute F1, IoU, precision, recall between predicted and ground truth masks."""
    pred_flat = (pred_mask > 0.5).flatten()
    gt_flat = (gt_mask > 0).flatten()
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, zero_division=0)
    intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0).sum()
    union = np.logical_or(pred_mask > 0.5, gt_mask > 0).sum()
    iou = intersection / union if union > 0 else 0.0
    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'IoU': iou
    }

def test_image(image_path, mask_path=None, model_path='sam_finetuned_final.pth', base_checkpoint='sam_vit_b_01ec64.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    try:
        sam = sam_model_registry["vit_b"](checkpoint=base_checkpoint).to(device)
        logging.info("Base SAM model loaded")
    except Exception as e:
        logging.error(f"Failed to load base model: {e}")
        raise

    try:
        checkpoint = torch.load(model_path, map_location=device)
        sam.mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])
        logging.info("Fine-tuned weights loaded")
    except Exception as e:
        logging.error(f"Failed to load fine-tuned model: {e}")
        raise

    sam.eval()
    logging.info("Model set to evaluation mode")

    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        logging.info(f"Loaded image: {image_path}")
    except Exception as e:
        logging.error(f"Failed to load image: {e}")
        raise

    transform = get_transforms()
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    prompts = np.array([[np.random.randint(64, 960), np.random.randint(64, 960)] for _ in range(5)], dtype=np.float32)
    prompts_tensor = torch.tensor(prompts, dtype=torch.float32).unsqueeze(0).to(device)
    labels = torch.ones((1, 5), device=device)
    logging.info(f"Generated prompts: {prompts}")

    with torch.no_grad():
        image_embeddings = sam.image_encoder(image_tensor)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(prompts_tensor, labels),
            boxes=None,
            masks=None
        )
        low_res_mask, _ = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

    pred_mask = torch.sigmoid(low_res_mask)
    pred_mask_np = pred_mask[0, 0].cpu().numpy()
    pred_mask_resized = cv2.resize(pred_mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred_mask_bin = (pred_mask_resized > 0.5) * 255

    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    pred_mask_path = os.path.join(output_dir, f"pred_{os.path.basename(image_path).rsplit('.', 1)[0]}.png")
    cv2.imwrite(pred_mask_path, pred_mask_bin)
    logging.info(f"Saved predicted mask: {pred_mask_path}")

    metrics = None
    gt_mask = None
    if mask_path:
        try:
            gt_mask = np.array(Image.open(mask_path).convert("L")) / 255.0
            logging.info(f"Loaded ground truth mask: {mask_path}")
            metrics = compute_metrics(pred_mask_resized, gt_mask)
            logging.info("Metrics:")
            for k, v in metrics.items():
                logging.info(f"{k}: {v:.4f}")
        except Exception as e:
            logging.error(f"Failed to load ground truth mask: {e}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3 if gt_mask is not None else 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3 if gt_mask is not None else 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask_resized, cmap='gray')
    plt.axis('off')

    if gt_mask is not None:
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth Mask")
        plt.imshow(gt_mask, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "image.jpg"
    mask_path = "org_mask.png"
    model_path = "sam_finetuned_final.pth"
    base_checkpoint = "sam_vit_b_01ec64.pth"
    test_image(image_path, mask_path, model_path, base_checkpoint)