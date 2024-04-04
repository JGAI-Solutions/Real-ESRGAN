# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models
import torch
from pickle import dump

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    torch.cuda.memory._record_memory_history(True)
    train_pipeline(root_path)
    # save a snapshot of the memory allocations
    s = torch.cuda.memory._snapshot()
    with open(f"snapshot.pickle", "wb") as f:
        dump(s, f)

    # tell CUDA to stop recording memory allocations now
    torch.cuda.memory._record_memory_history(False)
