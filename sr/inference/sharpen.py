from PIL import Image, ImageFilter


class Sharpener:
    """
    Post-processing sharpener using unsharp mask.
    Compensates for edge softening in L1-trained SR models.

    To remove: delete this file and the --sharpen block in upscale.py.
    """

    def __init__(self, radius: float = 1.5, percent: int = 150, threshold: int = 3):
        self.radius    = radius
        self.percent   = percent
        self.threshold = threshold

    def apply(self, image: Image.Image) -> Image.Image:
        y, cb, cr = image.convert('YCbCr').split()
        y_sharp = y.filter(
            ImageFilter.UnsharpMask(
                radius=self.radius,
                percent=self.percent,
                threshold=self.threshold,
            )
        )
        return Image.merge('YCbCr', (y_sharp, cb, cr)).convert('RGB')
