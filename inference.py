# ...existing code...
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from audiocraft.models.loaders import load_compression_model

from model.v2m_new import V2M
import argparse


class TorchAutocast:
    def __init__(self, enabled: bool, *args, **kwargs):
        self.autocast = torch.autocast(*args, **kwargs) if enabled else None

    def __enter__(self):
        if self.autocast is None:
            return
        try:
            self.autocast.__enter__()
        except RuntimeError:
            device = self.autocast.device
            dtype = self.autocast.fast_dtype
            raise RuntimeError(
                f"There was an error autocasting with dtype={dtype} device={device}\n"
                "If you are on the FAIR Cluster, you might need to use autocast_dtype=float16"
            )

    def __exit__(self, *args, **kwargs):
        if self.autocast is None:
            return
        self.autocast.__exit__(*args, **kwargs)


# Config / paths (preserved behavior)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
def parse_args():
    parser = argparse.ArgumentParser(description="Decoder for video-to-music generation")
    parser.add_argument("--model_path", type=str, 
                        required=True,
                        help="Path to the MusicGen compression model")
    parser.add_argument("--v2m_config_path", type=str,
                        default="./config/v2m.yaml",
                        help="Path to V2M config file")
    parser.add_argument("--v2m_checkpoint", type=str,
                        required=True,
                        help="Path to V2M checkpoint")
    parser.add_argument("--features_root", type=str,
                        required=True,
                        help="Root directory for video features")
    parser.add_argument("--out_sr", type=int, default=32000,
                        help="Output sample rate")
    parser.add_argument("--shot_max", type=int, default=10,
                        help="Maximum number of shots")
    parser.add_argument("--obj_max", type=int, default=5,
                        help="Maximum number of objects")
    parser.add_argument("--time_max", type=int, default=15,
                        help="Maximum time length")
    parser.add_argument(
        "--video_control",
        type=str,
        default="-1,-1,-1,1",
        help=(
            "Four-element CSV: shot_idx,obj_idx,feature,weight. "
            "shot_idx in [-1,shot_max] (-1 => all shots), "
            "obj_idx in [-1,obj_max] (-1 => all objects), "
            "feature in {semantic,color,move,area} or -1 for all, "
            "weight is float. Default='-1,-1,-1,1'"
        ),
    )
    parser.add_argument(
        "--music_control",
        type=str,
        default="-1,1",
        help=(
            "Two-element CSV: feature,weight. "
            "feature in {pitch,loudness,spectrum,chroma} or -1 for all, "
            "weight is float. Default='-1,1'"
        ),
    )
    return parser.parse_args()

def parse_video_control(s: str, shot_max: int, obj_max: int):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("video_control must have 4 comma-separated elements")
    try:
        shot_idx = int(parts[0])
        obj_idx = int(parts[1])
    except ValueError:
        raise ValueError("first two elements of video_control must be integers")
    feat_part = parts[2]
    if feat_part != "-1":
        feat = feat_part.lower()
        if feat not in ("semantic", "color", "move", "area"):
            raise ValueError("feature must be one of semantic/color/move/area or -1")
    else:
        feat = -1
    try:
        weight = float(parts[3])
    except ValueError:
        raise ValueError("fourth element of video_control must be a float")
    if not (-1 <= shot_idx <= shot_max and -1 <= obj_idx <= obj_max):
        raise ValueError(f"shot_idx must be in [-1,{shot_max}], obj_idx in [-1,{obj_max}]")
    return (shot_idx, obj_idx, feat, weight)

def parse_music_control(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("music_control must have 2 comma-separated elements")
    feat_part = parts[0]
    if feat_part != "-1":
        feat = feat_part.lower()
        if feat not in ("pitch", "loudness", "spectrum", "chroma"):
            raise ValueError("feature must be one of pitch/loudness/spectrum/chroma or -1")
    else:
        feat = -1
    try:
        weight = float(parts[1])
    except ValueError:
        raise ValueError("fourth element of video_control must be a float")
    
    return (feat, weight)

args = parse_args()
MODEL_PATH = args.model_path
V2M_CONFIG_PATH = args.v2m_config_path
V2M_CHECKPOINT = args.v2m_checkpoint
FEATURES_ROOT = Path(args.features_root)
OUT_SR = args.out_sr
SHOT_MAX = args.shot_max
OBJ_MAX = args.obj_max
TIME_MAX = args.time_max
VIDEO_CONTROL = parse_video_control(args.video_control, SHOT_MAX, OBJ_MAX)
MUSIC_CONTROL = parse_music_control(args.music_control)

def load_models(device: str):
    musicgen = load_compression_model(MODEL_PATH, device=device)
    with open(V2M_CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    v2m = V2M(config)
    ckpt = torch.load(V2M_CHECKPOINT, map_location=torch.device(device))
    v2m.load_state_dict(ckpt["model_state_dict"])
    v2m.to(device).eval()
    return v2m, musicgen


def load_shot_data(shot_dir: Path, shot_idx: int):
    pt_file = shot_dir / f"{shot_idx}.pt"
    npz_file = pt_file.with_suffix(".npz")
    sem = torch.load(pt_file)
    with np.load(npz_file) as data:
        hist = torch.from_numpy(data["hist.npy"])
        s_pos = torch.from_numpy(data["s_pos.npy"])
        pos = torch.from_numpy(data["position.npy"])
        area = torch.from_numpy(data["area.npy"])
    return sem, hist, s_pos, pos, area


def pad_object(tensor: torch.Tensor, obj_pad: int, pad_dims: Tuple[int, ...]):
    # pad_dims should be a tuple matching torch.nn.functional.pad expectation
    return F.pad(tensor, pad_dims, mode="constant", value=0)


def prepare_features(shot_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    semantics, colors, starts, moves, areas = [], [], [], [], []
    pre_second = 0
    shot_files = sorted([p for p in shot_dir.iterdir() if p.suffix == ".pt"], key=lambda p: int(p.stem))
    shot_number = len(shot_files)

    for i, pt in enumerate(shot_files):
        sem, hist, s_pos, pos, area = load_shot_data(shot_dir, int(pt.stem))

        obj_pad_num = max(0, OBJ_MAX - sem.shape[0])
        sem_pad = pad_object(sem, obj_pad_num, (0, 0, 0, obj_pad_num))
        semantics.append(sem_pad)

        # For arrays with extra dims: pad tuple length must match array dims
        object_padding = (0, 0, 0, 0, 0, obj_pad_num)
        time_pad_after = max(0, TIME_MAX - pre_second - pos.shape[2])
        
        # Truncate pos if pre_second exceeds TIME_MAX
        if pre_second >= TIME_MAX:
            pos = pos[:, :, :0]  # Empty tensor
        elif pre_second + pos.shape[2] > TIME_MAX:
            pos = pos[:, :, :TIME_MAX - pre_second]
        
        time_padding = (pre_second, time_pad_after, 0, 0, 0, obj_pad_num)

        colors.append(pad_object(hist, obj_pad_num, object_padding))
        starts.append(pad_object(s_pos, obj_pad_num, object_padding))
        moves.append(pad_object(pos, obj_pad_num, time_padding))
        areas.append(pad_object(area, obj_pad_num, (0, 0, 0, obj_pad_num)))

        pre_second += pos.shape[2]

    # Stack and apply shot-level padding
    semantics = torch.stack(semantics)
    colors = torch.stack(colors)
    starts = torch.stack(starts)
    moves = torch.stack(moves)
    areas = torch.stack(areas)

    shot_pad = max(0, SHOT_MAX - shot_number)
    shot_padding = (0, 0, 0, 0, 0, shot_pad)
    shot_padding1 = (0, 0, 0, 0, 0, 0, 0, shot_pad)

    semantics = F.pad(semantics, shot_padding, mode="constant", value=0).unsqueeze(0)
    colors = (
        F.pad(colors, shot_padding1, mode="constant", value=0)
        .reshape(-1, 3, 256)
        .unsqueeze(0)
    )
    starts = F.pad(starts, shot_padding1, mode="constant", value=0).reshape(-1, 2).unsqueeze(0)
    moves = (
        F.pad(moves, shot_padding1, mode="constant", value=0)
        .reshape(-1, 2, TIME_MAX)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    areas = F.pad(areas, shot_padding, mode="constant", value=0).reshape(-1, 1).unsqueeze(0)

    return semantics, colors, starts, moves, areas


def apply_feature_weight(semantics, colors, starts, moves, areas, spec):
    shot_idx, obj_idx, feat, weight = spec
    # semantics: (1, shots, objs, ...)
    # colors: (1, shots*objs, 3, 256)
    # starts: (1, shots*objs, 2)
    # moves: (1, TIME_MAX, shots*objs, 2)
    # areas: (1, shots*objs, 1)
    shots = semantics.shape[1]
    objs = OBJ_MAX
    shot_range = range(shots) if shot_idx == -1 else ([shot_idx] if 0 <= shot_idx < shots else [])
    obj_range = range(objs) if obj_idx == -1 else ([obj_idx] if 0 <= obj_idx < objs else [])

    if not shot_range or not obj_range:
        return semantics, colors, starts, moves, areas

    for s in shot_range:
        for o in obj_range:
            global_idx = s * objs + o
            if feat == -1 or feat == "semantic":
                semantics[0, s, o] = semantics[0, s, o] * weight
            if feat == -1 or feat == "color":
                colors[0, global_idx] = colors[0, global_idx] * weight
            if feat == -1:
                # when spec is -1 apply to starts/moves/areas as well
                starts[0, global_idx] = starts[0, global_idx] * weight
                moves[0, :, global_idx, :] = moves[0, :, global_idx, :] * weight
                areas[0, global_idx, :] = areas[0, global_idx, :] * weight
            elif feat == "move":
                moves[0, :, global_idx, :] = moves[0, :, global_idx, :] * weight
            elif feat == "area":
                areas[0, global_idx, :] = areas[0, global_idx, :] * weight
    return semantics, colors, starts, moves, areas


def process_number(number: str, v2m: V2M, musicgen, device: str):
    shot_dir = FEATURES_ROOT / number
    shot_number = int(len(list(shot_dir.iterdir())) / 2) 
    semantics, colors, starts, moves, areas = prepare_features(shot_dir)

    # apply video_control before moving to device
    semantics, colors, starts, moves, areas = apply_feature_weight(
        semantics, colors, starts, moves, areas, VIDEO_CONTROL
    )

    semantics = semantics.to(device)
    colors = colors.to(device)
    starts = starts.to(device)
    moves = moves.to(device)
    areas = areas.to(device)

    with torch.no_grad():
        with TorchAutocast(enabled=True, device_type="cuda", dtype=torch.float16):
            pitch, chroma, loudness, spectral, music_out = v2m.infer(semantics, colors, starts, moves, areas, MUSIC_CONTROL)
        token = music_out
        gen_audio = musicgen.decode(token, None)

    gen_audio = gen_audio.cpu()
    torchaudio.save(f"{number}.wav", gen_audio[0], OUT_SR)


def main():
    v2m, musicgen = load_models(DEVICE)
    # number_list can be replaced by reading a file or iterating directory
    number_list: List[str] = ["0102_0011"] # for batch processing
    for number in number_list:
        process_number(number, v2m, musicgen, DEVICE)


if __name__ == "__main__":
    main()