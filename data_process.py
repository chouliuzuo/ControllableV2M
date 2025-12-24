import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import cv2
import argparse
from VAEA.src.utils import *

from segment.script.amg import main
from segment.SegGPT.SegGPT_inference.seggpt_inference import prepare_model
from segment.SegGPT.SegGPT_inference.seggpt_engine import inference_video
from segment.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment.CLIP import clip

import torch
import argparse

import shutil
import numpy as np
from PIL import Image

def VAEA_process_video(video_path, tmp_folder = "./tmp"):
    os.makedirs(tmp_folder, exist_ok=True)
    for video_file in os.listdir(video_path):
        video_path_full = os.path.join(video_path, video_file)
        video_name = os.path.splitext(os.path.basename(video_path_full))[0]
        os.makedirs(os.path.join(tmp_folder, video_name), exist_ok=True)

        video = cv2.VideoCapture(video_path_full)
        fps = video.get(cv2.CAP_PROP_FPS)

        scenes = get_shot_boundaries(video_path_full)
        shots = get_shots(video,scenes,out_base64 = False)
        output_shots(shots, os.path.join(tmp_folder, video_name), fps)

        print(f"frame cnt of each shot: {[len(shot) for shot in shots]}")

def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
        "segment_number": args.seg_num,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def segment_process_video(input_dir, device, vit_checkpoint, vit_model_type, args, output_dir):
    args.output = output_dir
    print("Loading model...")
    model_type = "seggpt_vit_large_patch16_input896x448"
    ckpt_path = "./segment/SegGPT/SegGPT_inference/seggpt_vit_large.pth"
    model = prepare_model(ckpt_path, model_type, "instance").to(device)

    clip_model, preprocess = clip.load("ViT-L/14@336px", device=device, jit=False)

    sam = sam_model_registry[vit_model_type](checkpoint=vit_checkpoint)
    output_mode = "binary_mask"
    _ = sam.to(device=device)

    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    entries = sorted(os.listdir(input_dir))
    for files in entries:
        files_path = os.path.join(input_dir, files)
        if os.path.isdir(files_path):
            entries2  = sorted(os.listdir(files_path))
            for entry in entries2:
                if entry.endswith(".mp4") and os.path.isfile(os.path.join(files_path, entry)):
                    args.input = os.path.join(files_path, entry)
                    mask_list = main(generator, clip_model, preprocess, files, args)  # SAM
                    vid_name = os.path.basename(args.input)
                    
                    for i in range(len(mask_list)):
                        out_path = os.path.join(output_dir, files, "output_" + ".".join(vid_name.split(".")[:-1]) + str(i) + ".mp4")
                        base = os.path.basename(args.input)
                        base = os.path.splitext(base)[0]
                        np_path = os.path.join(output_dir, files, base)
                        np_path = os.path.join(np_path, str(i) + ".npz")
                        inference_video(model, device, args.input, 0, None, mask_list[i], out_path, np_path)

def merge_files(input_dir, output_dir, object_num=5):
    print("Merging files...")
    entries = sorted(os.listdir(input_dir))
    for files in entries:
        files_path = os.path.join(input_dir, files)
        if os.path.exists(os.path.join(output_dir, files)):
            continue
        if os.path.isdir(files_path):
            entries2  = sorted(os.listdir(files_path))
            for files2 in entries2:
                files_path2 = os.path.join(files_path, files2)
                if os.path.isdir(files_path2):
                    hist_cat = None
                    pos_cat = None
                    pt_cat = None
                    area_cat = None
                    for i in range(object_num):
                        if not os.path.exists(os.path.join(files_path2, str(i)+".npz")):
                            break
                        npz_data = np.load(os.path.join(files_path2, str(i)+".npz"))
                        hist_b = npz_data["hist_b"]
                        hist_b = hist_b.T
                        hist_g = npz_data["hist_g"]
                        hist_g = hist_g.T
                        hist_r = npz_data["hist_r"]
                        hist_r = hist_r.T
                        hist = np.concatenate((hist_b, hist_g, hist_r), axis=0).reshape(1, 3, 256)
                        position_x = npz_data["position_x"]
                        position_y = npz_data["position_y"]
                        position = np.vstack((position_x, position_y))
                        position = position.reshape(1, position.shape[0], position.shape[1])
                        area = np.array(npz_data["area"]).reshape(1, 1)
                        if hist_cat is None:
                            hist_cat = hist
                        else:
                            hist_cat = np.concatenate((hist_cat, hist), axis=0)
                        if pos_cat is None:
                            pos_cat = position
                        else:
                            pos_cat = np.concatenate((pos_cat, position), axis=0)
                        if area_cat is None:
                            area_cat = area
                        else:
                            area_cat = np.concatenate((area_cat, area), axis=0)
                    for i in range(object_num):
                        if not os.path.exists(os.path.join(files_path2, str(i)+".pt")):
                            break
                        pt_data = torch.load(os.path.join(files_path2, str(i)+".pt"))
                        if pt_cat is None:
                            pt_cat = pt_data
                        else:
                            pt_cat = torch.concat([pt_cat, pt_data], axis=0)
                    print(hist_cat.shape, pos_cat.shape, pt_cat.shape)
                    os.makedirs(os.path.join(output_dir, files), exist_ok=True)
                    total_area = area_cat.sum()
                    area_cat = area_cat / total_area
                    position = pos_cat
                    if position.shape[-1] == 0:
                        continue
                    img_path = os.path.join(files_path, '0', '0.png')
                    print(img_path)
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                    except Exception as e:
                        print(f"无法打开图片 {img_path}: {e}")
                    tmp = np.array([height, width])
                    s_pos = (position[:,:,0] / tmp - 0.5) * 2
                    s_pos = s_pos.reshape(position.shape[0], position.shape[1], 1)
                    position = position - position[:,:,0].reshape(position.shape[0], position.shape[1], 1)
                    np.savez(
                        os.path.join(output_dir, files, files2 +  ".npz"),
                        **{"area": area_cat, "hist": hist_cat, "s_pos": s_pos, "position": position},
                    )
                    torch.save(pt_cat, os.path.join(output_dir, files, files2 +  ".pt"))

video_path = "./examples"
tmp_folder = "./tmp"
seg_dir = "./segment_output"
output_dir = "./output"
device = "cuda" if torch.cuda.is_available() else "cpu"
vit_checkpoint = "./segment/images/sam_vit_h_4b8939.pth"
vit_model_type = "vit_h"
VAEA_process_video(video_path, tmp_folder)
parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)
parser.add_argument(
    "--seg_num", type=int, default=5, help="the number of segmentation"
)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="The device to run generation on."
)

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)
amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    # default=None,
    default=100,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)
if __name__ == "__main__":
    args = parser.parse_args()
    shutil.rmtree(seg_dir, ignore_errors=True)
    os.makedirs(tmp_folder, exist_ok=True)
    segment_process_video(tmp_folder, device, vit_checkpoint, vit_model_type, args, seg_dir)
    merge_files(seg_dir, output_dir)
    shutil.rmtree(tmp_folder, ignore_errors=True)
    shutil.rmtree(seg_dir, ignore_errors=True)