import argparse
from pathlib import Path
import random
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm


def generate_random_crop(image_size, crop_size_min):
    crop_size_max = min(image_size)
    crop_size = random.randint(crop_size_min, crop_size_max)

    max_x = image_size[0] - crop_size
    max_y = image_size[1] - crop_size

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    return (x, y, x + crop_size, y + crop_size)


def preprocess_images(image_path, output_dir, crop_size_min, lq_size, gt_size, num_crops=2, n_jobs=-1):
    image_path = Path(image_path)
    lq_output_dir = Path(str(output_dir) + str(lq_size))
    lq_output_dir.mkdir(parents=True, exist_ok=True)
    gt_output_dir = Path(str(output_dir) + str(gt_size))
    gt_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(image_path.iterdir())

    Parallel(n_jobs=n_jobs)(
        delayed(process_single_image)(
            image_path, lq_output_dir, gt_output_dir, crop_size_min, lq_size, gt_size, num_crops
        ) for image_path in tqdm(image_paths, desc='Processing Images'))


def save_image(image: Image.Image, size: int, output_path: Path):
    resized_image = image.resize((size, size), Image.ANTIALIAS)
    resized_image.save(output_path)


def make_lq_and_gt_pair(image: Image.Image, lq_output_path: Path, gt_output_path: Path, lq_size: int, gt_size: int):
    save_image(image, lq_size, lq_output_path)
    save_image(image, gt_size, gt_output_path)


def process_single_image(image_path: Path, lq_output_dir: Path, gt_output_dir: Path, crop_size_min, lq_size: int, gt_size: int, num_crops: int):
    image = Image.open(image_path)
    # Process and save original resized images and crops
    make_lq_and_gt_pair(
        image,
        lq_output_dir / image_path.name,
        gt_output_dir / image_path.name,
        lq_size,
        gt_size,
    )
    
    if min(image.size) > crop_size_min:
        for i in range(num_crops):
            crop_coordinates = generate_random_crop(image.size, crop_size_min)
            cropped_image = image.crop(crop_coordinates)
            image_name = f"{image_path.stem}_crop_{i}{image_path.suffix}"
            make_lq_and_gt_pair(
                cropped_image,
                lq_output_dir / image_name,
                gt_output_dir / image_name,
                lq_size,
                gt_size,
            )


def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess images by cropping and resizing')
    parser.add_argument('image_path', type=Path, help='Path to the source images directory')
    parser.add_argument('output_dir', type=Path, help='Output directory for processed images')
    parser.add_argument('--crop_size_min', type=int, default=128, help='Minimum crop size')
    parser.add_argument('--lq_size', type=int, default=64, help='LQ output resolution dimension')
    parser.add_argument('--gt_size', type=int, default=128, help='GT output resolution dimension')
    parser.add_argument('--num_crops', type=int, default=2, help='Number of crops per image')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs to run (default is -1, which uses all processors)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    preprocess_images(
        args.image_path,
        args.output_dir,
        args.crop_size_min,
        args.lq_size,
        args.gt_size,
        args.num_crops,
        args.n_jobs
    )
