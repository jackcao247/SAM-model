from flask import Flask, request, jsonify, render_template_string
import torch
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segment_anything import sam_model_registry, SamPredictor

app = Flask(__name__)

sam_predictor = None
unet_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_unet_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ])

def load_sam_model():
    global sam_predictor
    try:
        sam_finetuned_path = r'finetuned path'
        sam_base_path = r'pretrained'
        
        if not os.path.exists(sam_base_path):
            print("SAM base checkpoint not found")
            return False
            
        sam = sam_model_registry["vit_b"](checkpoint=sam_base_path)
        
        if os.path.exists(sam_finetuned_path):
            checkpoint = torch.load(sam_finetuned_path, map_location=device)
            sam.mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])
            print("SAM weights loaded")
        else:
            print("No weights for SAM")
        
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        print("SAM loaded")
        return True
        
    except Exception as e:
        print(f"SAM load failed {e}")
        return False

def load_unet_model():
    global unet_model
    try:
        unet_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        
        checkpoint_paths = [
            r'U-Net checkpoint'
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            return False
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            unet_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            unet_model.load_state_dict(checkpoint['state_dict'])
        else:
            unet_model.load_state_dict(checkpoint)
        
        unet_model.to(device)
        unet_model.eval()
        return True
        
    except Exception:
        return False

def predict_sam_mask(image_np):
    try:
        if sam_predictor is None:
            return None
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_np
        
        sam_predictor.set_image(image_rgb)
        
        h, w = image_rgb.shape[:2]
        center_point = np.array([[w//2, h//2]])
        center_label = np.array([1])
        
        masks, scores, logits = sam_predictor.predict(
            point_coords=center_point,
            point_labels=center_label,
            multimask_output=True,
        )
        
        best_mask = masks[np.argmax(scores)]
        return (best_mask * 255).astype(np.uint8)
        
    except Exception as e:
        print(f"SAM pred failed {e}")
        return None

def predict_unet_mask(image_np):
    try:
        if unet_model is None:
            print("U-Net failed")
            return None
        
        original_h, original_w = image_np.shape[:2]
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_np
        
        transform = get_unet_transforms()
        augmented = transform(image=image_rgb)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = unet_model(image_tensor)
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > 0.5).float()
        
        mask_256 = prediction.cpu().numpy()[0, 0]
        
        mask_resized = cv2.resize(mask_256, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        mask = (mask_resized * 255).astype(np.uint8)
        return mask
        
    except Exception as e:
        print(f"U-Net pred failed {e}")
        return None

def image_to_base64(image_np):
    if len(image_np.shape) == 2:
        pil_img = Image.fromarray(image_np, mode='L')
    else:
        pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    pil_img = Image.open(io.BytesIO(img_data))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

@app.route('/nails', methods=['POST'])
def predict_nails():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'No image'}), 400
        
        image_np = base64_to_image(data['image'])
        
        sam_mask = predict_sam_mask(image_np)
        unet_mask = predict_unet_mask(image_np)
        
        result = {}
        
        if sam_mask is not None:
            result['sam_mask'] = image_to_base64(sam_mask)
        else:
            result['sam_error'] = 'SAM pred failed'
        
        if unet_mask is not None:
            result['unet_mask'] = image_to_base64(unet_mask)
        else:
            result['unet_error'] = 'U-Net pred failed'
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'file' not in request.files:
            return jsonify({'No file dir'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'No file dir'}), 400
        
        image_bytes = file.read()
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if image_np is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        sam_mask = predict_sam_mask(image_np)
        unet_mask = predict_unet_mask(image_np)
        
        result = {
            'original_image': image_to_base64(image_np)
        }
        
        if sam_mask is not None:
            result['sam_mask'] = image_to_base64(sam_mask)
        
        if unet_mask is not None:
            result['unet_mask'] = image_to_base64(unet_mask)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_mask/<mask_type>')
def download_mask(mask_type):
    try:
        return jsonify({'Mask failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Nail Segmentation</title>
</head>
<body>
    <input type="file" id="fileInput" accept="image/*">
    <div id="results"></div>

    <script>
        document.getElementById('fileInput').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                const formData = new FormData();
                formData.append('file', e.target.files[0]);

                fetch('/process', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    let html = '';
                    
                    if (data.original_image) {
                        html += '<div><img src="data:image/png;base64,' + data.original_image + '" style="max-width:300px;"><div>Original</div></div>';
                    }
                    
                    if (data.sam_mask) {
                        html += '<div><img src="data:image/png;base64,' + data.sam_mask + '" style="max-width:300px;"><div>SAM</div><a href="data:image/png;base64,' + data.sam_mask + '" download="sam_mask.png">Download</a></div>';
                    }
                    
                    if (data.unet_mask) {
                        html += '<div><img src="data:image/png;base64,' + data.unet_mask + '" style="max-width:300px;"><div>U-Net</div><a href="data:image/png;base64,' + data.unet_mask + '" download="unet_mask.png">Download</a></div>';
                    }
                    
                    document.getElementById('results').innerHTML = html;
                })
                .catch(error => {
                    document.getElementById('results').innerHTML = '<div>Error: ' + error.message + '</div>';
                });
            }
        });
    </script>
</body>
</html>

    ''')

def initialize_models():
    sam_loaded = load_sam_model()
    print(f"SAM loaded")
    
    print("Loading U-Net model...")
    unet_loaded = load_unet_model()
    print(f"U-Net loaded")
    
    if not sam_loaded:
        print("SAM failed")
    if not unet_loaded:
        print("U-Net failed")

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
