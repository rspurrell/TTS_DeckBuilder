import cv2
import numpy as np
import sys
import os
import re
import argparse
from PIL import Image
from dataclasses import dataclass

#region classes
@dataclass
class CropRegion:
    x: int
    y: int
    width: int
    height: int

@dataclass
class ResizeTarget:
    width: int
    height: int
    gamma: float = 2.2

class DebugViewer:
    def __init__(self, max_width=800, max_height=800):
        self.zoom_scale = 1.0
        self.fit_mode = True
        self.max_width = max_width
        self.max_height = max_height

    def show(self, window_name, image):
        self.original = image
        display = self.get_display_image()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        h, w = display.shape[:2]
        cv2.resizeWindow(window_name, w, h)
        cv2.imshow(window_name, display)

    def get_display_image(self):
        h, w = self.original.shape[:2]
        if self.fit_mode:
            scale = min(self.max_width / w, self.max_height / h, 1.0)
        else:
            scale = self.zoom_scale
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(self.original, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def handle_key(self, key):
        if key in [ord('+'), ord('=')]:
            self.zoom_scale *= 1.2
            self.fit_mode = False
        elif key == ord('-'):
            self.zoom_scale /= 1.2
            self.fit_mode = False
        elif key in [ord('z'), ord('Z')]:
            self.fit_mode = not self.fit_mode
            if self.fit_mode:
                self.zoom_scale = 1.0
#endregion

#region debugging functions
def show_debug_images(original, contours, cropped, viewer_orig, viewer_crop):
    debug_img = original.copy()
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(debug_img, [box], 0, (255, 0, 0), 5) # (BGR) 5px blue line

    viewer_orig.show("Detected Regions", debug_img)
    viewer_crop.show("Cropped/Rotated", cropped)

def run_debug_viewers(original, cropped, count, contour):
    should_save = False
    viewer_orig = DebugViewer()
    viewer_crop = DebugViewer(400, 400)
    while True:
        print(f"Detected area: {cv2.contourArea(contour)}px")
        show_debug_images(original, [contour], cropped, viewer_orig, viewer_crop)
        key = cv2.waitKey(0)
        if key in [13, ord('\r')]:  # Enter key
            should_save = True
            break
        elif key in [ord('+'), ord('='), ord('-'), ord('z'), ord('Z')]:
            viewer_orig.handle_key(key)
            viewer_crop.handle_key(key)
        else:
            print(f"Skipped region {count}")
            break

    return should_save
#endregion

#region detect and straighten functions
def get_next_output_index(output_dir, prefix="output_", ext=".png"):
    """
    Scans the output directory and returns the next available index for saving images.
    """
    pattern = re.compile(rf"{re.escape(prefix)}(\d+){re.escape(ext)}$")
    max_index = 0
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
    return max_index + 1  # Next available number

def ensure_orientation(image, mode="auto"):
    if mode == "auto":
        return image

    h, w = image.shape[:2]
    if mode == "landscape" and h > w:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif mode == "portrait" and w > h:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def apply_final_crop(image, region: CropRegion):
    h_img, w_img = image.shape[:2]

    # Ensure crop region is within image bounds
    x = max(0, min(region.x, w_img - 1))
    y = max(0, min(region.y, h_img - 1))
    w = min(region.width, w_img - x)
    h = min(region.height, h_img - y)

    return image[y:y+h, x:x+w]

def resize_with_gamma(image, target: ResizeTarget):
    """
    High-quality resize using Pillow with gamma correction and LANCZOS resampling.
    """
    gamma = target.gamma

    # Convert BGR OpenCV image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to float32 in linear space (if gamma > 0)
    if gamma and gamma > 0:
        rgb_linear = np.power(rgb_image / 255.0, gamma)
    else:
        rgb_linear = rgb_image / 255.0

    # Convert to PIL image
    pil_image = Image.fromarray((rgb_linear * 255).astype(np.uint8), mode='RGB')

    # Resize using high-quality LANCZOS
    resized_pil = pil_image.resize((target.width, target.height), resample=Image.LANCZOS)

    # Convert back to numpy float for post-processing
    resized_np = np.asarray(resized_pil).astype(np.float32) / 255.0

    # Apply inverse gamma correction
    if gamma and gamma > 0:
        corrected = np.power(resized_np, 1.0 / gamma)
    else:
        corrected = resized_np

    # Convert back to OpenCV BGR 8-bit format
    corrected_bgr = cv2.cvtColor((corrected * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    return corrected_bgr

def find_contours(image, block_size=25, adaptive_threshold=15):
    """
    Preprocess the input image and return a list of contours that are likely
    to represent individual regions (e.g. photos on a scanned page).

    Args:
        image: Input BGR image (e.g. scanned page)

    Returns:
        List of 4-point approximated contours
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Better thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, adaptive_threshold
    )

    # Improve contour edge detection
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def order_points(pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def warp_from_contour(image, contour, min_area=10000):
    """
    Warps the image based on the contour and ensures orientation is either landscape or portrait,
    based on which dimension is larger.
    """
    area = cv2.contourArea(contour)
    if area < min_area: # Skip small artifacts
        return None

    rect = cv2.minAreaRect(contour)
    src_pts = cv2.boxPoints(rect)
    src_pts = order_points(np.array(src_pts, dtype="float32"))

    # Compute width and height based on distances between points
    (tl, tr, br, bl) = src_pts

    edge_b = np.linalg.norm(br - bl)  # bottom edge
    edge_t = np.linalg.norm(tr - tl)  # top edge
    edge_r = np.linalg.norm(tr - br)  # right edge
    edge_l = np.linalg.norm(tl - bl)  # left edge

    max_horizontal = int(max(edge_b, edge_t))
    max_vertical = int(max(edge_r, edge_l))

    # Set destination points for upright output
    dst_pts = np.array([
        [0, 0],
        [max_horizontal - 1, 0],
        [max_horizontal - 1, max_vertical - 1],
        [0, max_vertical - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (max_horizontal, max_vertical))
    return warped

def detect_and_straighten(image_path, output_dir, block_size, threshold, min_area, orientation, final_crop, resize_to, debug):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    contours = find_contours(img, block_size, threshold)

    regionCount = saved_count = skipped_count = 0
    start_index = get_next_output_index(output_dir)
    for cnt in contours:
        warped = warp_from_contour(img, cnt, min_area)
        if warped is None:
            continue

        rotated = ensure_orientation(warped, orientation)

        if final_crop:
            rotated = apply_final_crop(rotated, final_crop)

        if resize_to:
            rotated = resize_with_gamma(rotated, resize_to)

        should_save = True  # default to saving unless skipped during debug
        if debug:
            should_save = run_debug_viewers(img, rotated, regionCount, cnt)

        if should_save:
            output_path = os.path.join(output_dir, f"output_{start_index:04d}.png")
            cv2.imwrite(output_path, rotated)
            print(f"Saved: {output_path}")
            start_index += 1
            saved_count += 1

        regionCount += 1

    if saved_count == 0:
        print("No valid regions detected.")
    else:
        print(f"Done. {saved_count} images saved to {output_dir}")
        if debug:
            print(f"{skipped_count} images skipped.")
#endregion

def create_image_grid(source_dir, output_dir, grid_size, output_filename="combined_grid.jpg"):
    cols, rows = grid_size
    pattern = re.compile(r"output_(\d+)\.(jpg|png)$")

    # Get and sort output images
    files = sorted([
        f for f in os.listdir(source_dir)
        if pattern.match(f)
    ])

    if len(files) == 0:
        print("No images found for grid.")
        return

    images = [cv2.imread(os.path.join(source_dir, f)) for f in files]
    if not all(img is not None for img in images):
        raise ValueError("Failed to load one or more images for grid.")

    # Resize all images to same size
    h_min = min(img.shape[0] for img in images)
    w_min = min(img.shape[1] for img in images)
    resized = [cv2.resize(img, (w_min, h_min)) for img in images]

    # Pad with blanks if needed
    total_slots = cols * rows
    blank = np.zeros((h_min, w_min, 3), dtype=np.uint8)
    resized += [blank] * (total_slots - len(resized))

    # Create rows
    grid_rows = []
    for i in range(rows):
        row_imgs = resized[i * cols:(i + 1) * cols]
        row = np.hstack(row_imgs)
        grid_rows.append(row)

    # Stack rows
    grid_image = np.vstack(grid_rows)

    # Save
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, grid_image)
    print(f"Grid image saved: {output_path}")

#region argument parsing functions
def parse_crop_region(crop_str):
    try:
        parts = crop_str.split(",")
        if len(parts) != 3:
            raise ValueError
        x = int(parts[0])
        y = int(parts[1])
        w, h = map(int, parts[2].lower().split("x"))
        return CropRegion(x, y, w, h)
    except Exception:
        raise ValueError("Invalid crop format. Use format: 'x,y,WIDTHxHEIGHT' (e.g., 19,21,2042x1445)")

def parse_resize_target(resize_str, gamma=2.2):
    try:
        width, height = map(int, resize_str.lower().split("x"))
        return ResizeTarget(width, height, gamma)
    except Exception:
        raise ValueError("Invalid resize format. Use format: WIDTHxHEIGHT (e.g. 578x409)")

def parse_grid(grid_str):
    try:
        cols, rows = map(int, grid_str.lower().split("x"))
        return (cols, rows)
    except Exception:
        raise ValueError("Invalid grid format. Use format: COLSxROWS (e.g. 6x3)")
#endregion

if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser(description="Detect and straighten multiple images from a scanned photo.")
    parser.add_argument("source", help="Path to source image")
    parser.add_argument("output", help="Directory to save results")
    parser.add_argument("--min-area", type=int, default=10000,
        help="Minimum area (in pixels) for a detected contour to be considered valid (avoids artifacts)."
    )
    parser.add_argument("--adaptive-block-size", type=int, default=25,
        help="Block size for adaptive thresholding (must be odd and > 1)"
    )
    parser.add_argument( "--adaptive-threshold", type=int, default=15,
        help="Adjusts adaptive threshold sensitivity. Higher values make the detected image cleaner (reduces noise), but might lose fine detail. Lower values can preserve smaller or faint objects, but may introduce noise or false contours."
    )
    parser.add_argument("--orientation", choices=["landscape", "portrait", "auto"], default="auto",
        help="Preferred orientation: landscape, portrait, or auto (content-based)"
    )
    parser.add_argument("--final-crop", type=str, default=None,
        help="Crop final image to a region: 'x,y,WIDTHxHEIGHT' (e.g. '19,21,2042x1445')"
    )
    parser.add_argument("--resize-to",
        help="Resize output image to WIDTHxHEIGHT using bicubic resampling with gamma correction")
    parser.add_argument("--gamma", type=float, default=2.2,
        help="Gamma correction value for resizing (0 to disable, default=2.2). Only appied when --resize-to option is specified."
    )
    parser.add_argument("--grid",
        help="Combine all output images into a single image grid with format COLSxROWS (e.g. 6x3). Not compatible with any option other than output."
    )
    parser.add_argument("--debug", action="store_true", help="Show debug previews and allow image confirmation. Enter to save. +/-/z to zoom.")

    # parse arguments
    args = parser.parse_args()

    if args.adaptive_block_size and (args.adaptive_block_size & 1 == 0 or args.adaptive_block_size <= 1):
        raise ValueError("Block size for adaptive thresholding must be odd and > 1")

    # normalize arguments
    if args.final_crop:
        args.final_crop = parse_crop_region(args.final_crop)

    if args.resize_to:
        args.resize_to = parse_resize_target(args.resize_to, gamma=args.gamma)

    if args.grid:
        args.grid = parse_grid(args.grid)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # determine execution
    if args.grid:
        create_image_grid(args.source, args.output, args.grid)
    else:
        detect_and_straighten(args.source, args.output,
            args.adaptive_block_size, args.adaptive_threshold, args.min_area,
            args.orientation, args.final_crop, args.resize_to, args.debug
        )

    if args.debug:
        cv2.destroyAllWindows()
