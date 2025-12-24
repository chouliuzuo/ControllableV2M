import os
from script.amg import main
import argparse
from SegGPT.SegGPT_inference.seggpt_inference import prepare_model
from SegGPT.SegGPT_inference.seggpt_engine import inference_video


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from CLIP import clip

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


if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading model...")

    model_type = "seggpt_vit_large_patch16_input896x448"
    ckpt_path = "./SegGPT/SegGPT_inference/seggpt_vit_large.pth"
    model = prepare_model(ckpt_path, model_type, "instance").to(args.device)

    clip_model, preprocess = clip.load("ViT-L/14@336px", device=args.device, jit=False)

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    _ = sam.to(device=args.device)
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    input_dir = args.input
    # entries = sorted(os.listdir(input_dir))
    # for files in entries:
    #     files_path = os.path.join(input_dir, files)
    #     if os.path.isdir(files_path):
    #         entries2  = sorted(os.listdir(files_path))
            # for entry in entries2:
            #     if entry.endswith(".mp4") and os.path.isfile(os.path.join(files_path, entry)):
    import time
    # args.input = os.path.join(files_path, entry)
    args.input = '../tmp/0.mp4'
    files = '0430_0303'
    start_time = time.time()
    mask_list = main(generator, clip_model, preprocess, files, args)  # SAM
    vid_name = os.path.basename(args.input)
    
    for i in range(len(mask_list)):  # SAM 每个mask进行视频追踪
        out_path = os.path.join(args.output, files, "output_" + ".".join(vid_name.split(".")[:-1]) + str(i) + ".mp4")
        base = os.path.basename(args.input)
        base = os.path.splitext(base)[0]
        np_path = os.path.join(args.output, files, base)
        np_path = os.path.join(np_path, str(i) + ".npz")
        inference_video(model, args.device, args.input, 0, None, mask_list[i], out_path, np_path)
    end_time = time.time()
    print(f"Processed in {end_time - start_time:.2f} seconds")
