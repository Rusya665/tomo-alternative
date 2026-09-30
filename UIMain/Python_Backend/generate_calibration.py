"""
generate_calibration.py
=======================
Generate volumetric additive manufacturing (VAM / CAL) calibration patterns and
projector videos based on:
    Kelly et al. 2019, 'Volumetric additive manufacturing via tomographic reconstruction',
    Supplementary Information Fig. S2 and Section S6 ('Resin response calibration').

OpenCAL Filename Convention:
    OpenCAL reads the RPM directly from the video filename matching:
        r"_(\d+(?:\.\d+)?)rpm"
    e.g. calibration_gray_9rpm.mp4, calibration_white_9rpm.mp4.
    When loaded in OpenCAL via USB or PRINTS_DIR, OpenCAL automatically sets the
    motor speed to 9 RPM.

Two Videos Generated:
    1. calibration_gray_9rpm.mp4:
       10 circular spots with stepped intensities (100% down to 25%, matching
       Kelly et al. Fig. S2 intensity calibration).
    2. calibration_white_9rpm.mp4:
       10 circular spots all at 100% white for geometric, rotational axis, and
       threshold dose calibration.

Color & Projector Pipeline (Optoma ML1080):
    - Pure monochromatic blue channel [0, 0, g] (shuts off red & green laser diodes)
    - np.flipud (matches vamtoolbox bottom-origin pyglet projector texture)
    - 1080x1920 portrait resolution, 54 fps (standard for 9 RPM), 300s duration
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
import cv2
import imageio_ffmpeg


def generate_pattern_2d(
    width: int = 1080,
    height: int = 1920,
    num_dots: int = 10,
    dot_radius: int = 35,
    span_px: int = 1400,
    center_y: int = None,
    mode: str = "shades_of_gray",
    min_val: float = 0.25,
    max_val: float = 1.0,
):
    """Generate 2D grayscale base image g (uint8, shape (height, width))."""
    img = np.zeros((height, width), dtype=np.uint8)
    cx = width // 2

    if center_y is None:
        center_y = height // 2

    y_start = center_y - span_px / 2.0
    y_step = span_px / (num_dots - 1) if num_dots > 1 else 0

    if mode == "white":
        intensities = [255] * num_dots
    else:  # 'shades_of_gray'
        intensities = [int(round(255 * v)) for v in np.linspace(max_val, min_val, num_dots)]

    for i in range(num_dots):
        cy = int(round(y_start + i * y_step))
        val = int(intensities[i])
        cv2.circle(img, (cx, cy), dot_radius, val, -1, lineType=cv2.LINE_AA)

    return img, intensities


def apply_color_and_flip(g: np.ndarray, color_mode: str = "blue", flip: bool = True) -> np.ndarray:
    """Apply np.flipud and monochromatic blue [0, 0, g] or grey [g, g, g]."""
    frame_g = np.flipud(g) if flip else g

    if color_mode.lower() == "blue":
        zeros = np.zeros_like(frame_g)
        return np.stack([zeros, zeros, frame_g], axis=2)  # (H, W, 3) RGB
    else:
        return np.repeat(frame_g[:, :, None], 3, axis=2)  # (H, W, 3) RGB


def save_calibration_video(
    out_path: str,
    base_image: np.ndarray,
    color_mode: str = "blue",
    fps: float = 54.0,
    duration_s: float = 300.0,
    codec: str = "h264",
    flip: bool = True,
):
    """Encode calibration video via bundled imageio_ffmpeg stream-copy looping."""
    H, W = base_image.shape[:2]
    frame = apply_color_and_flip(base_image, color_mode=color_mode, flip=flip)

    codec_map = {"h265": "libx265", "h264": "libx264", "mp4v": "mpeg4"}
    ff_codec = codec_map.get(codec.lower(), "libx264")

    seg_frames = max(1, int(round(fps)))
    n_loops = max(1, int(round(duration_s)))

    seg_path = out_path + ".seg.mp4"
    writer = imageio_ffmpeg.write_frames(
        seg_path,
        (W, H),
        fps=fps,
        codec=ff_codec,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=2,
        quality=7,
    )
    writer.send(None)
    raw_frame = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
    for _ in range(seg_frames):
        writer.send(raw_frame)
    writer.close()

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            exe,
            "-y",
            "-stream_loop",
            str(n_loops - 1),
            "-i",
            seg_path,
            "-c",
            "copy",
            "-fflags",
            "+genpts",
            out_path,
        ],
        check=True,
        capture_output=True,
    )

    try:
        if os.path.exists(seg_path):
            os.remove(seg_path)
    except Exception:
        pass

    print(f"[CALIBRATION] Video saved: {out_path} ({W}x{H}, {fps} fps, {duration_s}s, {color_mode})")


def generate_opencal_calibration(tomo_dir: str, rpm: float = 9.0, fps: float = 54.0, duration_s: float = 300.0):
    """
    Generate exactly 2 videos conforming to OpenCAL naming convention:
        <name>_<rpm>rpm.mp4
    """
    os.makedirs(tomo_dir, exist_ok=True)
    img_dir = os.path.join(tomo_dir, "calibration_images")
    os.makedirs(img_dir, exist_ok=True)

    # Clean up any legacy video names in tomo_dir
    legacy = [
        "calibration_shades_of_gray_blue.mp4",
        "calibration_shades_of_gray_grey.mp4",
        "calibration_uniform_white_blue.mp4",
        "calibration_uniform_white_grey.mp4",
    ]
    for old in legacy:
        p = os.path.join(tomo_dir, old)
        if os.path.exists(p):
            os.remove(p)

    W, H = 1080, 1920

    # 1. Shades of gray pattern
    gray_img, _ = generate_pattern_2d(W, H, num_dots=10, dot_radius=35, span_px=1400, mode="shades_of_gray")
    # 2. Uniform white pattern
    white_img, _ = generate_pattern_2d(W, H, num_dots=10, dot_radius=35, span_px=1400, mode="white")

    rpm_tag = int(rpm) if rpm == int(rpm) else rpm

    # 1. Shades of gray video (OpenCAL filename)
    gray_vid_name = f"calibration_gray_{rpm_tag}rpm.mp4"
    gray_vid_path = os.path.join(tomo_dir, gray_vid_name)
    save_calibration_video(
        out_path=gray_vid_path,
        base_image=gray_img,
        color_mode="blue",
        fps=fps,
        duration_s=duration_s,
        codec="h264",
        flip=True,
    )

    # 2. White video (OpenCAL filename)
    white_vid_name = f"calibration_white_{rpm_tag}rpm.mp4"
    white_vid_path = os.path.join(tomo_dir, white_vid_name)
    save_calibration_video(
        out_path=white_vid_path,
        base_image=white_img,
        color_mode="blue",
        fps=fps,
        duration_s=duration_s,
        codec="h264",
        flip=True,
    )

    # Also copy to OpenCAL-alternative/prints if directory exists
    opencal_prints = Path.home() / "OpenCAL-alternative" / "prints"
    if opencal_prints.exists():
        import shutil
        shutil.copy2(gray_vid_path, opencal_prints / gray_vid_name)
        shutil.copy2(white_vid_path, opencal_prints / white_vid_name)
        print(f"[CALIBRATION] Copied videos to OpenCAL prints: {opencal_prints}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2 OpenCAL-compliant calibration videos")
    parser.add_argument("--tomo-dir", default=r"C:\Users\runiza\GoogleProjects\tomo-alternative")
    parser.add_argument("--rpm", type=float, default=9.0, help="RPM encoded in filename for OpenCAL (default: 9.0)")
    parser.add_argument("--fps", type=float, default=54.0, help="Video frame rate (default: 54.0)")
    parser.add_argument("--duration", type=float, default=300.0, help="Duration in seconds (default: 300.0)")
    args = parser.parse_args()

    print(f"Generating 2 calibration videos for OpenCAL ({args.rpm} RPM)...")
    generate_opencal_calibration(args.tomo_dir, rpm=args.rpm, fps=args.fps, duration_s=args.duration)
    print("Done!")
