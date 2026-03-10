import numpy as np
import argparse
import os
import glob
import threading
import torch
import cv2
import gc
import sys
import datetime
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
current_dir = os.path.dirname(os.path.abspath(__file__))
base_model_path = os.path.join(current_dir, "base_models")
if base_model_path not in sys.path:
    sys.path.append(base_model_path)

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from base_models.base_model import VGGTAdapter, Pi3Adapter, MapAnythingAdapter
from LoopModels import LoopDetector
from LoopModelDBoW import RetrievalDBow
from loop_utils import Sim3LoopOptimizer, load_config

def remove_duplicates(data_list):
    pass

class VGGTLong:
    def __init__(self, image_dir, save_dir, config):
        self.config = config
        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
