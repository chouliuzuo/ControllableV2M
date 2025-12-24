# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

import numpy as np
import argparse
import json
import os
from typing import Any, Dict, List

import torch
from PIL import Image

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--seg_num", type=int, default=10, help="the number of segmentation"
)
parser.add_argument(
    "--device", type=str, default="cuda", help="The device to run generation on."
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
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def write_masks_to_folder(
    masks: List[Dict[str, Any]],
    img: np.ndarray,
    path: str,
    num: int,
    clip_model,
    preprocess,
) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    masks = mask_filter([masks[j]["segmentation"] for j in range(len(masks))], num)
    mask_list = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, mask_data in enumerate(masks):
        mask = mask_data
        filename = f"{i}.png"
        new_mask = (
            np.concatenate([mask, mask, mask], axis=0)
            .reshape((3, mask.shape[0], mask.shape[1]))
            .transpose(1, 2, 0)
        )
        """segment的图片"""
        pic = new_mask * img
        pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
        # CLIP
        cv2.imwrite(os.path.join(path, filename), pic)
        image = (
            preprocess(Image.open(os.path.join(path, filename))).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
        torch.save(image_features, os.path.join(path, filename.replace("png", "pt")))

        mask_list.append(mask * 255)
        """color histgram"""
        hist_b, hist_g, hist_r = color(pic)
        hist_b[0] -= mask.reshape(-1).size - np.count_nonzero(mask)
        hist_g[0] -= mask.reshape(-1).size - np.count_nonzero(mask)
        hist_r[0] -= mask.reshape(-1).size - np.count_nonzero(mask)
        np.savez(
            os.path.join(path, filename.replace("png", "npz")),
            area = np.count_nonzero(new_mask),
            hist_b=hist_b,
            hist_g=hist_g,
            hist_r=hist_r,
        )
        # cv2.imwrite(os.path.join(path, filename), mask * 255)
    #     mask_metadata = [
    #         str(i),
    #         str(mask_data["area"]),
    #         *[str(x) for x in mask_data["bbox"]],
    #         *[str(x) for x in mask_data["point_coords"][0]],
    #         str(mask_data["predicted_iou"]),
    #         str(mask_data["stability_score"]),
    #         *[str(x) for x in mask_data["crop_box"]],
    #     ]
    #     row = ",".join(mask_metadata)
    #     metadata.append(row)
    # metadata_path = os.path.join(path, "metadata.csv")
    # with open(metadata_path, "w") as f:
    #     f.write("\n".join(metadata))

    return mask_list


def color(image):
    channels = cv2.split(image)

    # 设置颜色范围
    color_ranges = [0, 256]

    # 初始化直方图参数
    hist_size = [256]  # 每个维度的大小
    channels_id = [0]  # 每个通道的索引

    # 计算每个通道的直方图
    hist_b = cv2.calcHist([channels[0]], channels_id, None, hist_size, color_ranges)
    hist_g = cv2.calcHist([channels[1]], channels_id, None, hist_size, color_ranges)
    hist_r = cv2.calcHist([channels[2]], channels_id, None, hist_size, color_ranges)

    return hist_b, hist_g, hist_r


# def compare(matrix,array):
#     score = 0
#     matrix = matrix.reshape(-1)
#     for i in range(len(array)):
#         score += np.dot(array[i].reshape(-1),matrix)
#     score /= np.count_nonzero(matrix)
#     return score
def mask_filter(masks, num=5, simi=0.4):  # 取了面积最大的num块
    result = []
    position = []
    filter_pos = [0]
    sorted_1 = sorted(masks, key=np.count_nonzero, reverse=True)
    result.append(sorted_1[0])
    position.append(np.nonzero(sorted_1[0]))
    i = 1
    while i < len(sorted_1) and len(result) < num - 1:
        flag = True
        pre_position = np.nonzero(sorted_1[i])
        for j in range(len(position)):
            if sim(pre_position, position[j]) > simi:
                flag = False
                break
        if flag:
            filter_pos.append(i)
            result.append(sorted_1[i])
            position.append(pre_position)
        i += 1
    # while i < len(sorted_1):
    #     pre_position = np.nonzero(sorted_1[i])
    #     for j in range(len(position)):
    #         if sim(pre_position, position[j]) > simi:
    #             filter_pos.append(i)
    #             break
    #     i += 1
    a = [sorted_1[i] for i in range(len(sorted_1)) if i in filter_pos]
    result.append(np.logical_not(np.logical_or.reduce(a, axis=0)))
    return np.array(result)

def sim(pos1, pos2): #len(pos1) <= len(pos2)
    coords1 = set([(r, c) for r, c in zip(pos1[0], pos1[1])])
    coords2 = set([(r, c) for r, c in zip(pos2[0], pos2[1])])
    common_elements = coords1.intersection(coords2)  
    return len(common_elements) / len(coords1)
    

def main(generator, clip_model, preprocess, dir_name, args: argparse.Namespace) -> None:
    output_mode = "binary_mask"

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, dir_name), exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        if t.endswith("mp4"):
            cap = cv2.VideoCapture(t)
            if not cap.isOpened():
                print("Error: Could not open video.")
                exit()
            ret, image = cap.read()
            cap.release()
        else:
            image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = generator.generate(image)

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, dir_name, base)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=False)
            mask_list = write_masks_to_folder(
                masks, image, save_base, args.seg_num, clip_model, preprocess
            )
            return mask_list
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
