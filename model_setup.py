import numpy as np
import torch

def get_yolo_keypoints_from_pil(img_pil, yolo_model, conf_thresh=0.25):
    """
    Run YOLO pose on a PIL image and return normalized keypoints [K, 2].
    """
    # YOLO accepts numpy arrays
    img_np = np.array(img_pil)

    results = yolo_model.predict(
        source=img_np,
        save=False,
        verbose=False,
        conf=conf_thresh
    )
    if len(results) == 0:
        return None

    result = results[0]
    if len(result.boxes) == 0:
        return None

    boxes = result.boxes
    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
    idx = torch.argmax(areas).item()

    kpts = result.keypoints.xy[idx].cpu().numpy()
    return kpts


from transformers import AutoImageProcessor, AutoModel

@torch.inference_mode()
def get_dino_patch_features(img_pil, processor, dino_model, device, patch_size):
    inputs = processor(images=img_pil, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)  # [1, 3, H_dino, W_dino]
    _, _, H_dino, W_dino = pixel_values.shape

    num_patches_h = H_dino // patch_size
    num_patches_w = W_dino // patch_size

    outputs = dino_model(pixel_values)
    last_hidden = outputs.last_hidden_state  # [1, 1+reg+N_patches, D]

    num_register_tokens = getattr(dino_model.config, "num_register_tokens", 0)
    patch_tokens_flat = last_hidden[:, 1 + num_register_tokens:, :]  # [1, N_patches, D]

    patch_features = patch_tokens_flat.unflatten(1, (num_patches_h, num_patches_w))
    patch_features = patch_features[0]  # [H', W', D]

    return patch_features, (H_dino, W_dino)


import numpy as np

def kpts_to_patch_indices(kpts_xy_norm, H_dino, W_dino, patch_size):
    x_pix = kpts_xy_norm[:, 0] * W_dino
    y_pix = kpts_xy_norm[:, 1] * H_dino

    num_patches_h = H_dino // patch_size
    num_patches_w = W_dino // patch_size

    i = (y_pix // patch_size).astype(int)
    j = (x_pix // patch_size).astype(int)

    i = np.clip(i, 0, num_patches_h - 1)
    j = np.clip(j, 0, num_patches_w - 1)
    return i, j


from PIL import Image

@torch.inference_mode()
def compute_frame_embedding_from_pil(img_pil, yolo_model, dino_model, processor, patch_size, device):
    # 1) YOLO keypoints
    kpts_xy_norm = get_yolo_keypoints_from_pil(img_pil, yolo_model)
    if kpts_xy_norm is None:
        return None  # no person detected

    # 2) DINO patch features
    patch_features, (H_dino, W_dino) = get_dino_patch_features(
        img_pil, processor, dino_model, device, patch_size
    )
    D = patch_features.shape[-1]

    # 3) Map keypoints to patches
    i, j = kpts_to_patch_indices(kpts_xy_norm, H_dino, W_dino, patch_size)
    i_t = torch.from_numpy(i).long().to(patch_features.device)
    j_t = torch.from_numpy(j).long().to(patch_features.device)

    kpt_feats = patch_features[i_t, j_t, :]  # [K, D]

    # 4) Concat coords (same as training)
    kpts_xy_t = torch.from_numpy(kpts_xy_norm).float().to(kpt_feats.device)  # [K, 2]
    kpt_feats_plus = torch.cat([kpt_feats, kpts_xy_t], dim=-1)  # [K, D+2]

    # 5) Mean over joints
    frame_embed = kpt_feats_plus.mean(dim=0)  # [D+2]

    return frame_embed.cpu().numpy()


import torch.nn.functional as F

def predict_phase_from_pil(img_pil, yolo_model, dino_model, processor, patch_size, classifier_model, device):
    frame_embed = compute_frame_embedding_from_pil(
        img_pil, yolo_model, dino_model, processor, patch_size, device
    )
    if frame_embed is None:
        return None, None  # no detection

    x = torch.from_numpy(frame_embed).unsqueeze(0).to(device)

    classifier_model.eval()
    with torch.no_grad():
        logits = classifier_model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())

    label_name = "lowering" if pred == 0 else "pushing"
    return label_name, probs
