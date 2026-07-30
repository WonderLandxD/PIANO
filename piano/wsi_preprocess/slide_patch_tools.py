import os
import opensdpc
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2



# Mask generation by OTSU algorithm
def get_bg_mask(thumbnail, kernel_size=1):
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    ret, threshold = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)

    close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(threshold), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    return (image_open / 255.0).astype(np.uint8)


def generate_patch_coordinates(img_x, img_y, x_size, y_size, x_overlap, y_overlap, bg_mask, blank_TH):
    """Generate top-left patch coordinates in the level-0 reference frame.

    All size arguments (x_size, y_size, x_overlap, y_overlap) must already be
    expressed in level-0 pixels. Returned coordinates are level-0 pixels, which
    is the reference frame expected by `read_region`.
    """
    x_step = x_size - x_overlap
    y_step = y_size - y_overlap
    if x_step <= 0 or y_step <= 0:
        raise ValueError(f'Overlap must be smaller than patch size (x_step={x_step}, y_step={y_step})')

    bg_mask_height, bg_mask_width = bg_mask.shape[0], bg_mask.shape[1]

    n_x = int(np.floor((img_x - x_size) / x_step)) + 1
    n_y = int(np.floor((img_y - y_size) / y_step)) + 1

    coordinates = []
    for i in range(max(n_x, 0)):
        for j in range(max(n_y, 0)):
            x_start = i * x_step
            y_start = j * y_step
            mask = bg_mask[
                int(np.floor(y_start / img_y * bg_mask_height)):int(np.ceil((y_start + y_size) / img_y * bg_mask_height)),
                int(np.floor(x_start / img_x * bg_mask_width)):int(np.ceil((x_start + x_size) / img_x * bg_mask_width))
            ]
            if mask.size == 0:
                continue
            if np.sum(mask == 0) / mask.size < blank_TH:
                coordinates.append((x_start, y_start))
    return coordinates

def save_image(img, path, quality=95):
    img.save(path, format='JPEG', quality=quality, optimize=True)


# Nominal micrometres-per-pixel of each supported objective magnification.
MAGNIFICATION_TO_MPP = {'40x': 0.25, '20x': 0.5}


def get_base_mpp(slide):
    """Micrometres per pixel at level 0, or None when the slide does not say."""
    props = getattr(slide, 'properties', None)
    if props is not None:
        for key in ('openslide.mpp-x', 'openslide.mpp-y'):
            value = props.get(key)
            if value:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        objective = props.get('openslide.objective-power')
        if objective:
            try:
                return 10.0 / float(objective)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    # sdpc slides expose the scan magnification directly
    scan_magnification = getattr(slide, 'scan_magnification', None)
    if scan_magnification:
        try:
            return 10.0 / float(scan_magnification)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return None


def select_level_by_magnification(slide, magnification, tolerance, allow_downscale=False):
    """Pick the pyramid level that yields patches at `magnification`.

    Returns (level, read_scale, description). `read_scale` is how many level
    pixels must be read per output pixel: 1.0 means the level already matches
    the target mpp, >1 means the caller has to read a larger region and shrink
    it (only when `allow_downscale`). `level` is None when this slide cannot
    serve the requested magnification and should be skipped.
    """
    target_mpp = MAGNIFICATION_TO_MPP[magnification]
    base_mpp = get_base_mpp(slide)
    if base_mpp is None:
        return None, None, 'slide reports no mpp/objective-power'

    level_mpps = [base_mpp * slide.level_downsamples[lvl] for lvl in range(slide.level_count)]
    errors = [abs(mpp / target_mpp - 1) for mpp in level_mpps]
    best_level = int(np.argmin(errors))

    if errors[best_level] <= tolerance:
        return best_level, 1.0, f'{magnification} -> level {best_level} ({level_mpps[best_level]:.3f} um/px)'

    available = ', '.join(f'L{i}={mpp:.3f}' for i, mpp in enumerate(level_mpps))
    if not allow_downscale:
        return None, None, (f'no level within {tolerance:.0%} of {magnification} '
                            f'(target {target_mpp:.3f} um/px; available {available})')

    # Fall back to the finest level that is still coarser-or-equal in mpp than
    # the target, then shrink; going the other way would upsample.
    finer_levels = [lvl for lvl, mpp in enumerate(level_mpps) if mpp < target_mpp * (1 + tolerance)]
    if not finer_levels:
        return None, None, (f'cannot reach {magnification} even by downscaling '
                            f'(target {target_mpp:.3f} um/px; available {available})')

    source_level = max(finer_levels, key=lambda lvl: level_mpps[lvl])
    read_scale = target_mpp / level_mpps[source_level]
    return source_level, read_scale, (f'{magnification} -> level {source_level} '
                                      f'({level_mpps[source_level]:.3f} um/px) downscaled {read_scale:.2f}x')


def resize_patch(img, out_w, out_h):
    """Shrink a patch; BOX matches integer factors exactly and is much faster."""
    if img.size == (out_w, out_h):
        return img
    integer_factor = img.size[0] % out_w == 0 and img.size[1] % out_h == 0
    resample = Image.BOX if integer_factor else Image.LANCZOS
    return img.resize((out_w, out_h), resample)




def func_patching(args, pair_list, thread_id):
    total_slides = len(pair_list) 
    
    for item, pair_path in enumerate(pair_list):
        slide_path = pair_path[0]
        save_path = pair_path[1]

        try:
            slide = opensdpc.OpenSdpc(slide_path)
        except Exception as exc:
            print(f'[THREAD {thread_id}] Failed to open {slide_path}: {exc}')
            continue

        try:
            patch_level, read_scale, reason = select_level_by_magnification(
                slide, args.magnification, getattr(args, 'mpp_tolerance', 0.15),
                getattr(args, 'allow_downscale', False))
            if patch_level is None:
                print(f'[THREAD {thread_id}] Skip {slide_path}: {reason}')
                continue

            os.makedirs(save_path, exist_ok=True)

            thumbnail_level = min(max(slide.level_count - int(args.thumb_n), 0), slide.level_count - 1)
            thumbnail = np.array(slide.read_region((0, 0), thumbnail_level, slide.level_dimensions[thumbnail_level]).convert('RGB'))

            black_pixel = np.where((thumbnail[:, :, 0] < 50) & (thumbnail[:, :, 1] < 50) & (thumbnail[:, :, 2] < 50))
            thumbnail[black_pixel] = [255, 255, 255]
            bg_mask = get_bg_mask(thumbnail, kernel_size=args.kernel_size)

            # Region to read at `patch_level` for one output patch.
            read_w = int(round(args.patch_w * read_scale))
            read_h = int(round(args.patch_h * read_scale))

            # One output pixel spans this many level-0 pixels, which converts
            # the patch geometry into the level-0 frame used for tiling.
            level0_per_output = slide.level_downsamples[patch_level] * read_scale
            x_size = int(round(args.patch_w * level0_per_output))
            y_size = int(round(args.patch_h * level0_per_output))
            x_overlap = int(round(args.overlap_w * level0_per_output))
            y_overlap = int(round(args.overlap_h * level0_per_output))
            img_x, img_y = slide.level_dimensions[0]

            coordinates = generate_patch_coordinates(img_x, img_y, x_size, y_size, x_overlap, y_overlap, bg_mask, args.blank_TH)

            patch_progress = tqdm(total=len(coordinates), 
                                desc=f'THREAD {thread_id} Slide {item+1}/{total_slides}', 
                                position=thread_id, 
                                ncols=90,  
                                leave=False)

            jpg_quality = getattr(args, 'jpg_quality', 95)
            for idx, (x_start, y_start) in enumerate(coordinates):
                img = slide.read_region((x_start, y_start), patch_level, (read_w, read_h)).convert('RGB')
                img = resize_patch(img, args.patch_w, args.patch_h)
                save_image(img, os.path.join(save_path, f"no{idx:06d}_{x_start:09d}x_{y_start:09d}y.jpg"), quality=jpg_quality)
                patch_progress.update(1)

            patch_progress.close()

            # Written last so that its presence marks the slide as fully done.
            thumbnail_save_path = os.path.join(save_path, 'thumbnail/x20_thumbnail.jpg')
            os.makedirs(os.path.dirname(thumbnail_save_path), exist_ok=True)
            save_image(Image.fromarray(thumbnail), thumbnail_save_path)
        finally:
            try:
                slide.close()
            except Exception:
                pass
