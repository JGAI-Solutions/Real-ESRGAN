from PIL import Image, ImageFilter
from pathlib import Path
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed


def blur_image(img: Image.Image, radius=2) -> Image.Image:
    """Blur a single image.

    Parameters:
    - img (Image): Image object.
    - radius (int): Radius of the bluring kernel.

    Return:
    - img (Image): Blured image.
    """
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def resize_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize an image.
    
    Parameters:
    - img (Image): Image object.
    - size (tuple[int, int]): Desired size of the image.

    Return:
    - image (Image): Resized image.
    """
    return img.resize(size)

def process_image(img_path: Path, target_dir: Path, size: tuple[int, int]) -> None:
    """
    Process a single image and save it to the target directory.

    Parameters:
    - img_path (Path): Path of the image to be resized.
    - target_dir (Path): Directory where resized images will be saved.
    - size (tuple[int, int]): Tuple indicating the new size (width, height).
    """
    with Image.open(img_path) as img:
        img = resize_image(img, size)
        img = blur_image(img, radius=2)
        output_path = target_dir / img_path.name
        img.save(output_path)


def process_images_from_dir(source_dir: Path, target_dir: Path, size: tuple[int, int]=(512, 512), radius: int) -> None:
    """Resize all PNG images in the source directory and save them to the target directory with the specified size,
    using joblib for improved performance.

    Parameters:
    - source_dir (Path): Directory containing the original PNG images.
    - target_dir (Path): Directory where resized images will be saved.
    - size (tuple[int, int]): Tuple indicating the new size (width, height).
    - radius (int): Radius of the blurring kernel.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    images = list(source_dir.glob('*.png'))
    Parallel(n_jobs=-1)(delayed(resize_image)(img_path, target_dir, size, radius) for img_path in tqdm(images, desc="Processing images"))


def parse_size(size_str):
    """
    Parse the size string in the format WIDTHxHEIGHT into a tuple (width, height).

    Parameters:
    - size_str: A string representing the size in the format WIDTHxHEIGHT.

    Returns:
    A tuple (width, height).
    """
    width, height = map(int, size_str.lower().split('x'))
    return width, height


def main():
    parser = argparse.ArgumentParser(description="Prepare data for upscaler training. The data will be resized and processed to introduce noise to image.")
    parser.add_argument("source_dir", type=Path, help="Directory contains the original PNG images.")
    parser.add_argument("target_dir", type=Path, help="Directory where resized images will be saved.")
    parser.add_argument("--size", type=str, default="512x512", help="Size of the resized images in the format WIDTHxHEIGHT.")
    parser.add_argument("--radius", type=int, default=2, help="Radius of the gaussian bluring kernel")

    args = parser.parse_args()
    size = parse_size(args.size)
    process_images_from_dir(args.source_dir, args.target_dir, size=size, radius=args.radius)


if __name__ == "__main__":
    main()
