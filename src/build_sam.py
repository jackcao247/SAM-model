import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader
import logging
import time
from datetime import timedelta
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--start-epoch', type=int, default=0)
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sam_training_fallback.log', mode='a'),
        logging.StreamHandler()
    ]
)

def dice_loss(pred, target, batch_size, smooth=1e-5):
    pred = F.interpolate(pred, size=(1024, 1024), mode='bilinear', align_corners=False)
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def load_checkpoint(sam, optimizer, load_path, start_epoch_default):
    try:
        checkpoint = torch.load(load_path, map_location=device)
        sam.mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', start_epoch_default)
        logging.info(f"Checkpoint: {load_path}")
        return sam, optimizer, epoch
    except Exception as e:
        logging.error(f"Failed to load {load_path}: {e}")
        raise

def main():
    start_time = time.time()

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        sam = sam_model_registry["vit_b"](checkpoint="/content/drive/My Drive/SAM data/checkpoints/sam_vit_b_01ec64.pth").to(device)
        logging.info("Model loaded")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    sam.mask_decoder.train()
    logging.info("Image and prompt encoders frozen, mask decoder set to train")

    optimizer = optim.Adam(sam.mask_decoder.parameters(), lr=1e-4)

    if args.resume and os.path.exists(args.resume):
        logging.info(f"Checkpoint {args.resume}")
        sam, optimizer, start_epoch = load_checkpoint(sam, optimizer, args.resume, args.start_epoch)
        logging.info(f"Resuming from {start_epoch + 1}")
    else:
        start_epoch = args.start_epoch
        logging.info("Start")

    from data_loaders import get_loaders
    try:
        train_loader, valid_loader, test_loader = get_loaders()
        logging.info(f"Training batch: {len(train_loader)}, Validation batch: {len(valid_loader)}, Test batch: {len(test_loader)}")
    except Exception as e:
        logging.error(f"Failed to run data loaders: {e}")
        raise

    num_epochs = 6
    batch_times = []
    os.makedirs('/content/drive/My Drive/SAM data/models', exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        logging.info(f"\n*** Starting epoch {epoch + 1}/{num_epochs} ***")
        sam.train()
        train_loss = 0

        for batch_idx, (images, masks, prompts) in enumerate(train_loader):
            try:
                batch_start = time.time()
                images, masks = images.to(device), masks.to(device)
                
                logging.debug(f"Input - Images shape: {images.shape}, Masks shape: {masks.shape}, Prompts shape: {prompts.shape}")
                logging.info(f"*** Current Epoch: {epoch + 1}/{num_epochs} ***")
                logging.info(f"Training batch {batch_idx + 1}/{len(train_loader)}")
                
                prompts = prompts.to(device)
                batch_size = images.shape[0]
                num_points = prompts.shape[1]
                labels = torch.ones((batch_size, num_points), device=device)
                logging.debug(f"Prompts shape: {prompts.shape}, Labels shape: {labels.shape}, Batch size: {batch_size}, Num points: {num_points}")

                image_embeddings = sam.image_encoder(images)
                logging.debug(f"Image embeddings shape: {image_embeddings.shape}")
                
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=(prompts, labels),
                    boxes=None,
                    masks=None
                )
                logging.debug(f"Prompt encoder - Sparse embeddings shape: {sparse_embeddings.shape}, Dense embeddings shape: {dense_embeddings.shape}")
                
                embed_dim = 256
                expected_shape = (batch_size, num_points + 1, embed_dim)
                if sparse_embeddings.shape != expected_shape:
                    logging.error(f"Sparse embeddings shape {sparse_embeddings.shape}, expected {expected_shape}")
                    raise ValueError("Sparse embeddings shape mismatch")
                
                if dense_embeddings.shape != (batch_size, 256, 64, 64):
                    logging.error(f"Dense embeddings shape {dense_embeddings.shape}, expected ({batch_size}, 256, 64, 64)")
                    raise ValueError("Dense embeddings shape mismatch")

                low_res_masks_list = []
                iou_pred_list = []
                for i in range(batch_size):
                    img_emb = image_embeddings[i:i+1]
                    sparse_emb = sparse_embeddings[i:i+1]
                    dense_emb = dense_embeddings[i:i+1]
                    logging.debug(f"Processing image {i+1}/{batch_size} - img_emb: {img_emb.shape}, sparse_emb: {sparse_emb.shape}, dense_emb: {dense_emb.shape}")
                    
                    low_res_mask, iou_pred = sam.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False
                    )
                    if low_res_mask.shape[1] > 1:
                        logging.warning(f"Image {i+1}: Multiple masks detected: {low_res_mask.shape}. Selecting first mask.")
                        low_res_mask = low_res_mask[:, :1, :, :]
                    low_res_masks_list.append(low_res_mask)
                    iou_pred_list.append(iou_pred)
                
                low_res_masks = torch.cat(low_res_masks_list, dim=0)
                iou_pred = torch.cat(iou_pred_list, dim=0)
                
                logging.debug(f"Mask decoder output - Low res masks shape: {low_res_masks.shape}, IoU pred shape: {iou_pred.shape}")

                if low_res_masks.shape != (batch_size, 1, 256, 256):
                    logging.error(f"Low res masks shape {low_res_masks.shape}, expected ({batch_size}, 1, 256, 256)")
                    raise ValueError("Mask output shape mismatch")

                loss = dice_loss(low_res_masks, masks, batch_size)
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"Invalid loss value: {loss}")
                    raise ValueError("Invalid loss detected")
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                logging.info(f"Batch loss: {loss.item():.4f}")
                logging.info(f"Batch {batch_idx + 1} runtime: {batch_time:.2f} seconds")

                if len(batch_times) >= 5:
                    avg_batch_time = sum(batch_times) / len(batch_times)
                    batches_left = (len(train_loader) * (num_epochs - epoch)) - (batch_idx + 1)
                    time_left = avg_batch_time * batches_left
                    logging.info(f"Uptime: {timedelta(seconds=int(time.time() - start_time))}")
                    logging.info(f"Time left: {timedelta(seconds=int(time_left))}")

            except Exception as e:
                logging.error(f"Error in training batch {batch_idx + 1}: {e}")
                raise

        sam.eval()
        valid_loss = 0
        logging.info("Validating")
        with torch.no_grad():
            for batch_idx, (images, masks, prompts) in enumerate(valid_loader):
                try:
                    images, masks = images.to(device), masks.to(device)
                    prompts = prompts.to(device)
                    batch_size = images.shape[0]
                    num_points = prompts.shape[1]
                    labels = torch.ones((batch_size, num_points), device=device)

                    logging.info(f"*** Current Epoch: {epoch + 1}/{num_epochs} ***")
                    logging.info(f"Validation batch {batch_idx + 1}/{len(valid_loader)}")

                    image_embeddings = sam.image_encoder(images)
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=(prompts, labels),
                        boxes=None,
                        masks=None
                    )
                    
                    embed_dim = 256
                    expected_shape = (batch_size, num_points + 1, embed_dim)
                    if sparse_embeddings.shape != expected_shape:
                        logging.error(f"Validation: Sparse embeddings shape {sparse_embeddings.shape}, expected {expected_shape}")
                        raise ValueError("Sparse embeddings shape mismatch")

                    if dense_embeddings.shape != (batch_size, 256, 64, 64):
                        logging.error(f"Validation: Dense embeddings shape {dense_embeddings.shape}, expected ({batch_size}, 256, 64, 64)")
                        raise ValueError("Dense embeddings shape mismatch")

                    low_res_masks_list = []
                    iou_pred_list = []
                    for i in range(batch_size):
                        img_emb = image_embeddings[i:i+1]
                        sparse_emb = sparse_embeddings[i:i+1]
                        dense_emb = dense_embeddings[i:i+1]
                        low_res_mask, iou_pred = sam.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=False
                        )
                        if low_res_mask.shape[1] > 1:
                            logging.warning(f"Validation: Image {i+1}: Multiple masks detected: {low_res_mask.shape}. Selecting first mask.")
                        low_res_mask = low_res_mask[:, :1, :, :]
                        low_res_masks_list.append(low_res_mask)
                        iou_pred_list.append(iou_pred)
                    
                    low_res_masks = torch.cat(low_res_masks_list, dim=0)
                    iou_pred = torch.cat(iou_pred_list, dim=0)

                    if low_res_masks.shape != (batch_size, 1, 256, 256):
                        logging.error(f"Validation: Low res masks shape {low_res_masks.shape}, expected ({batch_size}, 1, 256, 256)")
                        raise ValueError("Mask output shape mismatch")

                    loss = dice_loss(low_res_masks, masks, batch_size)
                    valid_loss += loss.item()
                    logging.info(f"Batch loss: {loss.item():.4f}")

                    if (batch_idx + 1) % 10 == 0:
                        logging.info(f"Validation batch {batch_idx + 1}/{len(valid_loader)}, Current average loss: {valid_loss / (batch_idx + 1):.4f}")
                except Exception as e:
                    logging.error(f"Error in validation batch {batch_idx + 1}: {e}")
                    raise

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Valid Loss: {valid_loss / len(valid_loader):.4f}")
        logging.info(f"Epoch runtime: {timedelta(seconds=int(time.time() - epoch_start))}")

        checkpoint_path = f'/content/drive/My Drive/SAM data/models/sam_finetuned_epoch_{epoch + 1}.pth'
        try:
            torch.save({
                'epoch': epoch,
                'mask_decoder_state_dict': sam.mask_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            raise

    logging.info("Saving final model")
    try:
        torch.save({
            'epoch': num_epochs - 1,
            'mask_decoder_state_dict': sam.mask_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, '/content/drive/My Drive/SAM data/models/sam_finetuned_final.pth')
    except Exception as e:
        logging.error(f"Failed to save final model: {e}")
    logging.info(f"Total runtime: {timedelta(seconds=int(time.time() - start_time))}")
    logging.info("Complete")

if __name__ == '__main__':
    main()