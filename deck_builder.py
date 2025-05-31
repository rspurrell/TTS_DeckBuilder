import cv2
import numpy as np
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
    def __init__(self, max_width=1000, max_height=800):
        self.zoom_scale = 1.0
        self.fit_mode = True
        self.max_width = max_width
        self.max_height = max_height

    def show(self, window_name, image):
        self.original = image
        display = self.get_display_image()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display)

    def get_display_image(self):
        if self.fit_mode:
            h, w = self.original.shape[:2]
            scale = min(self.max_width / w, self.max_height / h, 1.0)
        else:
            scale = self.zoom_scale
        new_w = max(1, int(self.original.shape[1] * scale))
        new_h = max(1, int(self.original.shape[0] * scale))
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
        cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)

    viewer_orig.show("Detected Regions", debug_img)
    viewer_crop.show("Cropped/Rotated", cropped)

def resize_for_display(image, max_width=1000, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)  # Only downscale
    return cv2.resize(image, (int(w * scale), int(h * scale)))

def run_debug_viewers(original, cropped, count, contour):
    should_save = False
    viewer_orig = DebugViewer()
    viewer_crop = DebugViewer()
    while True:
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

#region separate and straighten functions
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

def auto_rotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    if lines is None:
        return image  # No lines found

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)

    # Filter for mostly horizontal/vertical lines
    angles = [a for a in angles if -60 <= a <= 60 or abs(a) >= 85]

    if len(angles) < 5:
        return image  # Not enough lines to trust the angle

    median_angle = np.median(angles)

    # Snap to nearest 90°
    snapped_angle = round(median_angle / 90) * 90
    snapped_angle = snapped_angle % 360  # Normalize

    if snapped_angle == 0:
        return image  # Already properly oriented

    # Rotate image
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), snapped_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def detect_content_orientation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # horizontal edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # vertical edges

    sum_x = np.sum(np.abs(sobel_x))
    sum_y = np.sum(np.abs(sobel_y))

    return "landscape" if sum_x >= sum_y else "portrait"

def ensure_orientation(image, mode="landscape"):
    h, w = image.shape[:2]

    if mode == "auto":
        mode = detect_content_orientation(image)

    is_landscape = w >= h
    should_be_landscape = (mode == "landscape")

    if is_landscape != should_be_landscape:
        # Rotate to match desired orientation
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
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

def find_contours(image):
    """
    Preprocess the input image and return a list of contours that are likely
    to represent individual regions (e.g. photos on a scanned page).

    Args:
        image: Input BGR image (e.g. scanned page)

    Returns:
        List of 4-point approximated contours
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #edged = cv2.Canny(blur, 50, 200)

    # Better thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 15
    )

    # Improve contour edge detection
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def order_points(pts):
    # Order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and diff to sort
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]        # top-left
    rect[2] = pts[np.argmax(s)]        # bottom-right
    rect[1] = pts[np.argmin(diff)]     # top-right
    rect[3] = pts[np.argmax(diff)]     # bottom-left

    return rect

def warp_from_contour(image, contour, min_area=10000):
    """
    Warps the image based on the contour and ensures orientation is either landscape or portrait,
    based on which dimension is larger.
    """
    area = cv2.contourArea(contour)
    if area < min_area:
        return None

    rect = cv2.minAreaRect(contour)
    print(rect)
    width, height = rect[1]

    print(f"Original: {width},{height}")

    # Sanity check
    if width == 0 or height == 0:
        return None

    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")
    print(box)

    # Determine if the detected rectangle is closer to portrait or landscape
    is_landscape = width >= height

    # Define destination size with corrected orientation
    target_width = int(max(width, height))
    target_height = int(min(width, height))

    if not is_landscape:
        target_width, target_height = target_height, target_width  # swap for portrait

    print(f"Target: {target_width},{target_height}")
    dst_pts = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype="float32")

    # Order the source box points to match destination layout
    src_pts = order_points(box)

    # If the original rectangle is closer to portrait, rotate destination to match
    if not is_landscape:
        dst_pts = np.array([
            [0, target_height - 1],
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1]
        ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))

    return warped

def detect_and_straighten(image_path, output_dir, orientation, final_crop, resize_to, debug):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    contours = find_contours(img)

    regionCount = 0
    skipped_count = 0
    saved_count = 0
    start_index = get_next_output_index(output_dir)
    for cnt in contours:
        warped = warp_from_contour(img, cnt)
        if warped is None:
            continue

        rotated = warped
        #rotated = auto_rotate(warped)
        #rotated = ensure_orientation(rotated, orientation)

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

def create_image_grid(output_dir, grid_size, output_filename="combined_grid.jpg"):
    cols, rows = grid_size
    pattern = re.compile(r"output_(\d+)\.(jpg|png)$")

    # Get and sort output images
    files = sorted([
        f for f in os.listdir(output_dir)
        if pattern.match(f)
    ])

    if len(files) == 0:
        print("No images found for grid.")
        return

    images = [cv2.imread(os.path.join(output_dir, f)) for f in files]
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
    parser.add_argument("input", help="Path to input scanned image")
    parser.add_argument("output", help="Directory to save results")
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
        help="Combine all output images into a single image grid with format COLSxROWS (e.g. 6x3)"
    )
    parser.add_argument("--debug", action="store_true", help="Show debug previews and allow image confirmation")

    # parse arguments
    args = parser.parse_args()

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
        create_image_grid(args.output, args.grid)
    else:
        detect_and_straighten(args.input, args.output, args.orientation, args.final_crop, args.resize_to, args.debug)

    if args.debug:
        cv2.destroyAllWindows()
