import numpy as np
import argparse
import os
import glob
import threading
import torch
import cv2
import gc
import sys
from tqdm.auto import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
current_dir = os.path.dirname(os.path.abspath(__file__))
base_model_path = os.path.join(current_dir, "base_models")
if base_model_path not in sys.path:
    sys.path.append(base_model_path)

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")