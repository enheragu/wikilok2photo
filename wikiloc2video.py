#!/usr/bin/env python3
# encoding: utf-8

import os
import io
import re
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from wikiloc2photo import COLOR, TEXT_COLOR, IMGH_DIRECTION_DEFAULT, UPPER_MARGIN, fetch_dynamic_html, extract_static_data, extract_track_points


def parse_args():
    parser = argparse.ArgumentParser(description="Extract track data from a Wikiloc URL and overlay it on an image")
    parser.add_argument("video", type=str, help="Path to the video file.")
    parser.add_argument("url", type=str, help="URL to the Wikiloc track.")
    parser.add_argument("--img_direction", choices=['horizontal', 'vertical'], default=IMGH_DIRECTION_DEFAULT, help="Direction of the image layout")
    return parser.parse_args()

"""
    Interpolates a given x-y plot to have N points.
"""
def interpolate_points(x_points, y_points, n_points):
    x_points = np.array(x_points)
    y_points = np.array(y_points)

    interpolator = interp1d(x_points, y_points, kind="linear")
    new_x = np.linspace(x_points.min(), x_points.max(), n_points)
    new_y = interpolator(new_x)
    
    return new_x, new_y

def process_video(video_path, track_points, output_path, static_data):
    print(f"Processing {video_path} video.")
    cap = cv2.VideoCapture(video_path)
    print(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Total frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    if not cap.isOpened():
        print("Error: Could not open video file.")
        cap.release()
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Match number of points with number of frames
    n_points = frame_count
    x = [point[0] for point in track_points[1:-1]]
    y = [point[1] for point in track_points[1:-1]]

    proportion = width / x[-1]
    upper_margin_value = height*(1-UPPER_MARGIN)
    current_range_y = np.max(y) - np.min(y)
    # Use 2 thirds of space for track data
    y_scale_factor = max(1,(height*UPPER_MARGIN*0.6)/current_range_y)
    scaled_x = [point*proportion for point in x]
    scaled_y = np.array(y)*y_scale_factor+upper_margin_value
    
    scaled_x, scaled_y = interpolate_points(scaled_x, scaled_y, n_points)

    for frame_idx in tqdm(range(frame_count), desc="Processing frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.axis("off")
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))

        current_x = scaled_x[:frame_idx + 1]
        current_y = scaled_y[:frame_idx + 1]
        ax.plot(current_x, current_y, color=COLOR, linewidth=2)
        ax.fill_between(current_x, current_y, height, color=COLOR, alpha=0.75)

        # Bigger right point
        if frame_idx < len(current_x):
            ax.add_patch(Circle((current_x[-1], current_y[-1]), radius=2, color=TEXT_COLOR))

        ax.imshow(frame, interpolation='nearest') # Display the image.

        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, format="png", dpi=300, transparent=True)
        image_data = image_buffer.getvalue()
        image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image_np.shape[1] != width or image_np.shape[0] != height:
            image_np = cv2.resize(image_np, (width, height))  # Redimensionar si es necesario

        plt.close(fig)

        plt.close(fig)
        out.write(image_np)

    cap.release()
    out.release()
    print(f"Video stored in {output_path}")



def main():
    args = parse_args()
    html_content = fetch_dynamic_html(args.url)
    static_data = extract_static_data(html_content)

    svg_match = re.search(r'<svg.*?elevation-profile-svg.*?>(.*?)</svg>', html_content, re.DOTALL)
    if svg_match:
        svg_content = svg_match.group(1)
        track_points = extract_track_points(svg_content)
    else:
        track_points = []

    os.makedirs("./output", exist_ok = True)
    video_name, video_extension = os.path.splitext(os.path.basename(args.video))
    output_path = f"./output/{video_name}{video_extension}"
    process_video(args.video, track_points, output_path, static_data)

if __name__ == "__main__":
    main()