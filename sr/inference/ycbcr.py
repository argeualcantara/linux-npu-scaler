from PIL import Image


def split_ycbcr(img: Image.Image):
    """
    PIL Image → Y, Cb, Cr PIL images via PIL's built-in YCbCr conversion.
    Using PIL's native split is the most reliable approach and matches
    the ONNX Model Zoo preprocessing exactly.
    """
    ycbcr = img.convert('YCbCr')
    y, cb, cr = ycbcr.split()
    return y, cb, cr


def merge_ycbcr(y: Image.Image, cb: Image.Image, cr: Image.Image) -> Image.Image:
    """
    Merge Y_sr + Cb_bicubic + Cr_bicubic → RGB.
    Cb and Cr are bicubic-upscaled to match Y_sr size.
    """
    target_size = y.size
    cb_up = cb.resize(target_size, Image.BICUBIC)
    cr_up = cr.resize(target_size, Image.BICUBIC)
    return Image.merge('YCbCr', [y, cb_up, cr_up]).convert('RGB')
