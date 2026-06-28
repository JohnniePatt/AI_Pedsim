import numpy as np
from PIL import Image


BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def image_tensor_to_uint8(tensor_chw):
    arr = tensor_chw.detach().cpu().numpy() if hasattr(tensor_chw, "detach") else np.asarray(tensor_chw)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def save_triptych_sample(image_a, pred_density, real_b, out_path):
    a = Image.fromarray(image_tensor_to_uint8(image_a)).convert("RGB")
    p = Image.fromarray(image_tensor_to_uint8(pred_density)).convert("RGB")
    r = Image.fromarray(image_tensor_to_uint8(real_b)).convert("RGB")
    w, h = a.size
    canvas = Image.new("RGB", (w * 3, h), (0, 0, 0))
    canvas.paste(a, (0, 0))
    canvas.paste(p, (w, 0))
    canvas.paste(r, (w * 2, 0))
    canvas.save(out_path)


def tensor_to_pil(tensor_chw, out_w=None, out_h=None):
    arr = image_tensor_to_uint8(tensor_chw)
    mode = "L" if arr.ndim == 2 else "RGB"
    img = Image.fromarray(arr, mode=mode).convert("RGB")
    if out_w and out_h:
        img = img.resize((int(out_w), int(out_h)), BILINEAR)
    return img
