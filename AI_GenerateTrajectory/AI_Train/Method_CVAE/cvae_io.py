import numpy as np
from PIL import Image


def denorm_image_to_uint8(img_m11):
    arr = ((img_m11 + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def save_triptych_sample(image_a, pred_b, real_b, out_path):
    a = Image.fromarray(denorm_image_to_uint8(image_a))
    p = Image.fromarray(denorm_image_to_uint8(pred_b))
    r = Image.fromarray(denorm_image_to_uint8(real_b))

    w, h = a.size
    canvas = Image.new("RGB", (w * 3, h), (0, 0, 0))
    canvas.paste(a, (0, 0))
    canvas.paste(p, (w, 0))
    canvas.paste(r, (w * 2, 0))
    canvas.save(out_path)


def denorm_to_pil(image_m11, out_w, out_h):
    arr = ((image_m11 + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize((int(out_w), int(out_h)), Image.LANCZOS)
