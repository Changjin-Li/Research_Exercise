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
from loop_utils import Sim3LoopOptimizer, load_config, process_loop_list, weighted_align_point_maps

def remove_duplicates(data_list):
    """data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]"""
    seen = {}
    result = []
    for item in data_list:
        if item[0] == item[2]: continue
        key = (item[0], item[2])
        if key not in seen.keys():
            seen[key] = item
            result.append(item)
    return result

def extract_p2_k_matrix(calib_path):
    """from calib.txt get K  (kitti)"""
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith('P2:'):
                values = line.split(':')[1].split()
                values = [float(v) for v in values]
                p2_matrix = np.array(values).reshape(3, 4)
                k_matrix = p2_matrix[:3, :3]
                return k_matrix, p2_matrix

    raise ValueError("P2 not found in calibration file.")

class LongSeqResult:
    def __init__(self):
        self.combined_extrinsics = []
        self.combined_intrinsics = []
        self.combined_depth_maps = []
        self.combined_depth_confs = []
        self.combined_world_points = []
        self.combined_world_points_confs = []
        self.all_camera_poses = []
        self.all_camera_intrinsics = []


class VGGTLong:
    def __init__(self, image_dir, save_dir, config):
        self.config = config
        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False
        self.useDBoW = self.config['Model']['useDBoW']
        self.img_dir = image_dir
        self.img_list = None
        self.delete_temp_files = self.config['Model']['delete_temp_files']
        self.output_dir = save_dir
        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        self.all_camera_poses = []
        self.all_camera_intrinsics = []

        if self.config['Weights']['model'] == 'VGGT':
            self.model = VGGTAdapter(self.config)
        elif self.config['Weights']['model'] == 'Pi3':
            self.model = Pi3Adapter(self.config)
        elif self.config['Weights']['model'] == 'MapAnything':
            self.model = MapAnythingAdapter(self.config)
        else:
            raise ValueError(f"Unsupported model: {self.config['Weights']['model']}. ")

        self.sky_seg_session = None
        self.chunk_indices = None   # [(begin_idx, end_idx), ...]
        self.loop_list = []         # e.g. [(1584, 139), ...]
        self.loop_optimizer = Sim3LoopOptimizer(self.config)
        self.sim3_list = []         # [(s [1,], R [3,3], T [3,]), ...]
        self.loop_sim3_list = []    # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]
        self.loop_predict_list = []
        self.loop_enable = self.config['Model']['loop_enable']

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBow(self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(image_dir=image_dir, output=loop_info_save_path, config=self.config)
        print("init done.")

    def get_loop_pairs(self):
        if self.useDBoW: # DBoW2
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    # gray to rgb
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = image_ori   # (height, width, 3)
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(thresh=self.config['Loop']['DBoW']['thresh'], num_repeat=self.config['Loop']['DBoW']['num_repeat'])

                if cands is not None:
                    (i, j) = cands # e.g. cands = (812, 67)
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)
                self.retrieval.save_up_to(frame_id)

        else: # DINOv2
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        predictions = self.model.infer_chunk(chunk_image_paths)
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"
        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = predictions['extrinsics']
            intrinsics = predictions['intrinsics']
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))
        predictions['depth'] = np.squeeze(predictions['depth'])

        np.save(save_path, predictions)

        return predictions if is_loop or range_2 is not None else None

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))

        for chunk_idx in range(len(self.chunk_indices)):
            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)-1}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()    # 释放当前 GPU 上由 PyTorch 的缓存分配器持有的未使用的缓存内存

        if self.loop_enable:
            print('Loop SIM(3) estimating...')
            loop_results = process_loop_list(self.chunk_indices, self.loop_list, int(self.config['Model']['loop_chunk_size'] / 2))
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(range_1=item[1], range_2=item[3], is_loop=True)
                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)
        print(f"Processing {len(self.img_list)} images in {num_chunks} chunks of size {self.chunk_size} with {self.overlap} overlap")

        del self.model
        torch.cuda.empty_cache()

        print("Aligning all the chunks...")
        for chunk_idx in range(len(self.chunk_indices)-1):
            print(f"Aligning {chunk_idx} and {chunk_idx + 1} (Total {len(self.chunk_indices) - 1})")
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx + 1}.npy"), allow_pickle=True).item()
            point_map1 = chunk_data1['world_points'][-self.overlap:]
            point_map2 = chunk_data2['world_points'][:self.overlap]
            conf1 = chunk_data1['world_points_conf'][-self.overlap:]
            conf2 = chunk_data2['world_points_conf'][:self.overlap]

            mask = None
            if chunk_data1['mask'] is not None:
                mask1 = chunk_data1['mask'][-self.overlap:]
                mask2 = chunk_data2['mask'][:self.overlap]
                mask = mask1.unsqueeze() & mask2.unsqueeze()

            conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            s, R, t = weighted_align_point_maps(point_map1, conf1, point_map2, conf2, mask, conf_threshold, self.config)
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)
            self.sim3_list.append((s, R, t))

        if self.loop_enable:
            pass # TODO: loop align

        print('Done.')

    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')) + glob.glob(os.path.join(self.img_dir, '*.png')))
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images.")

        if self.loop_enable:
            self.get_loop_pairs()
            if self.useDBoW:
                self.retrieval.close()
                gc.collect()
            else:
                del self.loop_detector

        torch.cuda.empty_cache()
        print('Loading model...')
        self.model.load()

        if self.config['Model']['calib']:
            calib_path = Path(self.img_dir).parent / 'calib.txt'
            k, p2_matrix = extract_p2_k_matrix(calib_path)
            self.model.k = k

        self.process_long_sequence()

    def save_camera_poses(self):
        """
        Save camera poses from all chunks to txt and ply files.
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers.
        - ply file: Camera poses visualized as points with different colors for each chunk.
        """
        chunk_colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],    # Dark Red
            [0, 128, 0],    # Dark Green
            [0, 0, 128],    # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")
        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            all_poses[idx] = first_chunk_extrinsics[i]
            if first_chunk_intrinsics is not None:
                all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[chunk_idx - 1]     # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.
            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t
            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                c2w = chunk_extrinsics[i]
                transform_c2w = S @ c2w     # Be aware of the left multiplication!
                transform_c2w[:3, :3] /= s  # Normalize rotation.
                all_poses[idx] = transform_c2w
                if chunk_intrinsics is not None:
                    all_intrinsics[idx] = chunk_intrinsics[i]

        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')
        print(f"Camera poses saved to {poses_path}.")

        if all_intrinsics[0] is not None:
            intrinsics_path = os.path.join(self.output_dir, 'intrinsic.txt')
            with open(intrinsics_path, 'w') as f:
                for intrinsic in all_intrinsics:
                    fx = intrinsic[0, 0]
                    fy = intrinsic[1, 1]
                    cx = intrinsic[0, 2]
                    cy = intrinsic[1, 2]
                    f.write(f'{fx} {fy} {cx} {cy}\n')
            print(f"Camera intrinsics saved to {intrinsics_path}.")

        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:  # Write PLY header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f'{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n')
        print(f"Camera poses visualization saved to {ply_path}.")

    def close(self):
        """
        Clean up temporary files and calculate reclaimed disk space.
        This method deletes all temporary files generated during processing from three directories:
        - Unaligned results
        - Aligned results
        - Loop results
        ~50 GiB for 4500-frame KITTI 00, ~35 GiB for 2700-frame KITTI 05, or ~5 GiB for 300-frame short seq.
        """
        