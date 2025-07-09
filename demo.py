import argparse
import cv2
import logging
import os
import time  # Add time import for total processing time measurement

import torch
import yt_dlp
from mivolo.data.data_reader import get_all_files, get_input_type, InputType
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

_logger = logging.getLogger("inference")


def get_direct_video_url(video_url):
    ydl_opts = {
        "format": "bestvideo",
        "quiet": True,  # Suppress terminal output (remove this line if you want to see the log)
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)

        if "url" in info_dict:
            direct_url = info_dict["url"]
            resolution = (info_dict["width"], info_dict["height"])
            fps = info_dict["fps"]
            yid = info_dict["id"]
            return direct_url, resolution, fps, yid

    return None, None, None, None


def get_local_video_info(vid_uri):
    cap = cv2.VideoCapture(vid_uri)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source {vid_uri}")
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return res, fps


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--input", type=str, default=None, required=True, help="image file or folder with images")
    parser.add_argument("--output", type=str, default=None, required=True, help="folder for output results")
    parser.add_argument("--detector-weights", type=str, default=None, required=True, help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")

    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"
    )

    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")

    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Number of frames to skip between processed frames. 0 means process every frame.",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort.yaml",
        help="Tracker config file. Options: 'bytetrack.yaml', 'botsort.yaml'",
    )
    parser.add_argument(
        "--temporal-smoothing",
        action="store_true",
        default=False,
        help="Apply temporal smoothing to age/gender predictions for each ID",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=10,
        help="Number of frames to use for temporal smoothing (default: 10)",
    )
    parser.add_argument(
        "--global-smoothing",
        action="store_true",
        default=False,
        help="Apply global temporal smoothing across all frames (gives one final prediction per track). Note: incompatible with --confidence-based",
    )
    parser.add_argument(
        "--confidence-based",
        action="store_true",
        default=False,
        help="Use confidence-based selection: keep only the highest confidence prediction per track. Takes priority over --global-smoothing",
    )
    parser.add_argument(
        "--min-detection-conf",
        type=float,
        default=0.5,
        help="Minimum YOLO detection confidence for confidence-based selection (default: 0.5)",
    )
    parser.add_argument(
        "--min-gender-conf",
        type=float,
        default=0.6,
        help="Minimum gender prediction confidence for confidence-based selection (default: 0.6)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=False,
        help="Save tracking results to JSON file with demographics and locations",
    )
    parser.add_argument(
        "--json-output-path",
        type=str,
        default=None,
        help="Path to save JSON results (default: output/results.json)",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        default=True,
        help="Save cropped images of detected persons (default: True)",
    )
    parser.add_argument(
        "--no-save-crops",
        action="store_true",
        default=False,
        help="Disable saving cropped images (overrides --save-crops)",
    )
    parser.add_argument(
        "--crops-dir",
        type=str,
        default="crops",
        help="Directory name for saving cropped images (default: crops)",
    )

    return parser


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)

    # Set default JSON output path if saving JSON but no path specified
    if args.save_json and not args.json_output_path:
        args.json_output_path = os.path.join(args.output, "results.json")
    
    # Handle no-save-crops flag
    if args.no_save_crops:
        args.save_crops = False

    predictor = Predictor(args, verbose=True)

    input_type = get_input_type(args.input)

    if input_type == InputType.Video or input_type == InputType.VideoStream:
        if not args.draw:
            raise ValueError("Video processing is only supported with --draw flag. No other way to visualize results.")

        if "youtube" in args.input:
            args.input, res, fps, yid = get_direct_video_url(args.input)
            if not args.input:
                raise ValueError(f"Failed to get direct video url {args.input}")
            outfilename = os.path.join(args.output, f"out_{yid}.avi")
        else:
            bname = os.path.splitext(os.path.basename(args.input))[0]
            outfilename = os.path.join(args.output, f"out_{bname}.avi")
            res, fps = get_local_video_info(args.input)

        if args.draw:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(outfilename, fourcc, fps, res)
            _logger.info(f"Saving result to {outfilename}..")

        start_time = time.time()
        frame_count = 0
        for (
            detected_objects_history,
            frame,
        ) in predictor.recognize_video(args.input, skip_frames=args.skip_frames):
            frame_count += 1
            if args.draw:
                out.write(frame)
        end_time = time.time()
        total_processing_time = end_time - start_time
        avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        
        _logger.info(f"Video processing completed!")
        _logger.info(f"Total frames processed: {frame_count}")
        _logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
        _logger.info(f"Average processing FPS: {avg_fps:.2f}")
        
        if args.draw:
            out.release()
            _logger.info(f"Output video saved to: {outfilename}")

    elif input_type == InputType.Image:
        image_files = get_all_files(args.input) if os.path.isdir(args.input) else [args.input]

        for img_p in image_files:

            img = cv2.imread(img_p)
            detected_objects, out_im = predictor.recognize(img)

            if args.draw:
                bname = os.path.splitext(os.path.basename(img_p))[0]
                filename = os.path.join(args.output, f"out_{bname}.jpg")
                cv2.imwrite(filename, out_im)
                _logger.info(f"Saved result to {filename}")


if __name__ == "__main__":
    main()
