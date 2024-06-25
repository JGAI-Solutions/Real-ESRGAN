## Environment Installation

1. install python 3.8
2. `python -m venv .venv`
3. `source .venv/bin/activate`
2. install dependencies.
```
pip install -r requirements.txt
python setup.py develop
```
---

## ⚡ Quick Inference

### Python script

1. You can use X4 model for **arbitrary output size** with the argument `outscale`. The program will further perform cheap resize operation after the Real-ESRGAN output.

```console
Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile --model_path path/to/model [options]...

A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --model_path path/to/model --outscale 3.5 --face_enhance

  -h                   show this help
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  --model_path         Path to checkpoint of trained model.
  -s, --outscale       The final upsampling scale of the image. Default: 4
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --face_enhance       Whether to use GFPGAN to enhance face. Default: False
  --fp32               Use fp32 precision during inference. Default: fp16 (half precision).
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
  --ds --downscale     Downscale images before upscaling. By this technique you can introduce smoothing of pixels which can be helpful for generated images of poor quality
```

## Training

Steps to run training from the scratch.

1. Paste to experiments/pretrained_models directory `RealESRGAN_x4plus_netD.pth` (descriminator) and `RealESRGAN_x4plus.pth` (generator) models initial weights.
2. Indicate gpu ids that should be used during training (e.g. `export CUDA_VISIBLE_DEVICES=0,2`)
3. activate environment if you haven't by `source .venv/bin/activate`
4. setup paths to lq data and gt data if they are different. TO GENERATE DATA PLEASE READ SECTION [DATA PREPROCESSING]()
5. Run script.
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata_128_to_512.yml
```

## Data Preparation

1. Provide large scale data with a resolution (1024 or more).
2. Run data preparation script with minimum crop size (crop_size_min) equal 512, low quality (lq_size) outcome images size equal 128, ground truth (gt_size) outcome images size equal 512.
```
usage: prepare_data.py [-h] [--crop_size_min CROP_SIZE_MIN] [--lq_size LQ_SIZE] [--gt_size GT_SIZE] [--num_crops NUM_CROPS]
                       [--n_jobs N_JOBS]
                       image_path output_dir

Preprocess images by cropping and resizing

positional arguments:
  image_path            Path to the source images directory
  output_dir            Output directory for processed images

optional arguments:
  -h, --help            show this help message and exit
  --crop_size_min CROP_SIZE_MIN
                        Minimum crop size
  --lq_size LQ_SIZE     LQ output resolution dimension
  --gt_size GT_SIZE     GT output resolution dimension
  --num_crops NUM_CROPS
                        Number of crops per image
  --n_jobs N_JOBS       Number of parallel jobs to run (default is -1, which uses all processors)
```

e.g.
```
python data/prepare_data.py path/to/large/scale/data ../data/processed/upscaler/train --crop_size_min 512 --lq_size 128 --gt_size 512 --num_crops 2 --n_jobs 12
```
