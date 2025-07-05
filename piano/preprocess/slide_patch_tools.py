# Three backends are supported for generating patches


import os

# Set environment variables for large images BEFORE importing cv2 and PIL
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2,40))  # Very large limit for WSI

try:
    import opensdpc
except:
    import openslide
    OPENSDPC_AVAILABLE = False
    print('Opensdpc not available. Changing to openslide backend.')


import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# Import pyvips for alternative WSI processing
try:
    import pyvips
    PYVIPS_AVAILABLE = True
except ImportError:
    PYVIPS_AVAILABLE = False
    print("Warning: pyvips not available.")

# Increase PIL's image size limit for large WSI images
Image.MAX_IMAGE_PIXELS = None  # Remove PIL decompression bomb protection for WSI processing



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
    x_size_minus_overlap = x_size - x_overlap
    y_size_minus_overlap = y_size - y_overlap
    bg_mask_height, bg_mask_width = bg_mask.shape[0], bg_mask.shape[1]

    coordinates = []
    for i in range(int(np.floor((img_x - x_size) / x_size_minus_overlap + 1))):
        for j in range(int(np.floor((img_y - y_size) / y_size_minus_overlap + 1))):
            mask = bg_mask[
                int(np.floor(j * y_size_minus_overlap / img_y * bg_mask_height)):int(np.ceil((j * y_size_minus_overlap + y_size) / img_y * bg_mask_height)),
                int(np.floor(i * x_size_minus_overlap / img_x * bg_mask_width)):int(np.ceil((i * x_size_minus_overlap + x_size) / img_x * bg_mask_width))
            ]
            if np.sum(mask == 0) / mask.size < blank_TH:
                coordinates.append((i, j))
    return coordinates


def padding_image(image_array, patch_size):
    """padding an image on the right and bottom with [255, 255, 255]"""
    height, width = image_array.shape[:2]

    right = int(np.ceil(width / patch_size) * patch_size - width)
    bottom = int(np.ceil(height / patch_size) * patch_size - height)

    # padding
    image_array = cv2.copyMakeBorder(image_array, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return image_array


def is_deprecated(image_array, blank_rate):
    """whether to deprecate the patches with blank ratio greater than max_blank_ratio"""
    blank_num = np.sum(image_array == (255, 255, 255)) / 3
    height, width = image_array.shape[:2]
    if blank_num / (height * width) >= blank_rate:
        return True
    else:
        return False


def is_jpg_format(file_path):
    """Check if file is in JPG format"""
    return file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.PNG'))


def draw_annotation_thumbnail(thumbnail, bg_mask, patch_positions, thumbnail_save_path):
    """Draw annotated thumbnail showing cropping regions and patch positions"""
    # Create annotated thumbnail copy
    anno_thumbnail = thumbnail.copy()
    
    # Draw background mask contours (green for valid regions, blue for hollow regions)
    if bg_mask is not None:
        # Find contours
        contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw outer contours (green)
        cv2.drawContours(anno_thumbnail, contours, -1, (0, 255, 0), 2)
        
        # Find internal hollow contours (blue)
        contours_internal, _ = cv2.findContours(1 - bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(anno_thumbnail, contours_internal, -1, (255, 0, 0), 1)
    
    # Draw patch positions (red boxes) and index numbers
    for idx, (x, y, w, h) in enumerate(patch_positions):
        # Draw red box
        cv2.rectangle(anno_thumbnail, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        # Add patch index number (red text, small font)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        text = str(idx)
        
        # Calculate text size to determine position
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Place text at top-left corner inside the box
        text_x = x + 2
        text_y = y + text_height + 2
        
        # Ensure text doesn't exceed box boundaries
        if text_x + text_width <= x + w and text_y <= y + h:
            cv2.putText(anno_thumbnail, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
    
    # Save annotated thumbnail
    anno_thumbnail_path = os.path.join(os.path.dirname(thumbnail_save_path), 'anno_thumbnail.jpg')
    save_image(anno_thumbnail, anno_thumbnail_path)


def save_image(img, path):
    if isinstance(img, np.ndarray):
        # Convert BGR to RGB if it's a numpy array from cv2
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        else:
            pil_img = Image.fromarray(img)
        pil_img.save(path, 'JPEG', quality=100)
    else:
        img.save(path, 'JPEG', quality=100)


def process_jpg_wsi(slide_path, save_path, args, patch_progress):
    """Process JPG format WSI using PIL for large image support"""
    
    # print(f"Opening large JPG image: {slide_path}")
    
    # Load the image once and keep it in memory for processing
    big_image_pil = Image.open(slide_path)
    if big_image_pil.mode != 'RGB':
        big_image_pil = big_image_pil.convert('RGB')
    
    original_width, original_height = big_image_pil.size
    # print(f"Image size: {original_width}x{original_height} pixels")
    
    # Create thumbnail for background mask generation
    max_thumb_size = 1000
    if max(original_height, original_width) > max_thumb_size:
        scale = max_thumb_size / max(original_height, original_width)
        thumb_width = int(original_width * scale)
        thumb_height = int(original_height * scale)
        thumbnail_pil = big_image_pil.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
    else:
        thumbnail_pil = big_image_pil.copy()
        scale = 1.0
    
    # Convert PIL thumbnail to numpy array for cv2 processing (RGB format)
    thumbnail = np.array(thumbnail_pil)
    
    # Process black pixels (thumbnail is now in RGB format)
    black_pixel = np.where((thumbnail[:, :, 0] < 50) & (thumbnail[:, :, 1] < 50) & (thumbnail[:, :, 2] < 50))
    thumbnail[black_pixel] = [255, 255, 255]
    
    # Convert RGB to BGR for cv2 operations in get_bg_mask
    thumbnail_bgr = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
    
    # Generate background mask
    bg_mask = get_bg_mask(thumbnail_bgr, kernel_size=args.kernel_size)
    
    # Calculate patch parameters (no overlapping for JPG format)
    x_size = args.patch_w
    y_size = args.patch_h
    x_overlap = 0
    y_overlap = 0
    
    # Generate valid patch coordinates
    coordinates = generate_patch_coordinates(original_width, original_height, x_size, y_size, x_overlap, y_overlap, bg_mask, args.blank_TH)
    
    # Update progress bar
    patch_progress.total = len(coordinates)
    patch_progress.refresh()
    
    # print(f"Starting to extract {len(coordinates)} patches...")
    
    # Collect patch position information for annotation
    patch_positions = []
    
    # Process patches - now using the already loaded image
    patch_count = 0
    for idx, (i, j) in enumerate(coordinates):
        x_start = int(i * x_size)
        y_start = int(j * y_size)
        
        # Extract patch from already loaded image (much faster!)
        x_end = min(x_start + x_size, original_width)
        y_end = min(y_start + y_size, original_height)
        patch_box = (x_start, y_start, x_end, y_end)
        small_image_pil = big_image_pil.crop(patch_box)
        
        # Pad if necessary
        if small_image_pil.size != (x_size, y_size):
            padded_image = Image.new('RGB', (x_size, y_size), (255, 255, 255))
            padded_image.paste(small_image_pil, (0, 0))
            small_image_pil = padded_image
        
        # Save patch
        patch_path = os.path.join(save_path, f"no{idx:06d}_{x_start:09d}x_{y_start:09d}y.jpg")
        small_image_pil.save(patch_path, 'JPEG', quality=100)
        
        # Record patch position (scaled to thumbnail coordinates)
        patch_x = int(x_start * scale)
        patch_y = int(y_start * scale)
        patch_w = int(x_size * scale)
        patch_h = int(y_size * scale)
        patch_positions.append((patch_x, patch_y, patch_w, patch_h))
        
        patch_count += 1
        patch_progress.update(1)
    
    # Close the big image to free memory
    big_image_pil.close()
    
    # print(f"Extracted {patch_count} patches, saving thumbnails...")
    
    # Save regular thumbnail (same as opensdpc format)
    thumbnail_save_path = os.path.join(save_path, 'thumbnail/x20_thumbnail.jpg')
    os.makedirs(os.path.dirname(thumbnail_save_path), exist_ok=True)
    x20_thumbnail = Image.fromarray(thumbnail)
    save_image(x20_thumbnail, thumbnail_save_path)
    
    # Draw annotated thumbnail (same as opensdpc format)
    draw_annotation_thumbnail(thumbnail_bgr, bg_mask, patch_positions, thumbnail_save_path)
    
    # print(f"JPG processing completed successfully!")
    return patch_count


def process_pyvips_wsi(slide_path, save_path, args, patch_progress):
    """Process WSI format using pyvips for large image support"""
    
    if not PYVIPS_AVAILABLE:
        raise ImportError("pyvips is not available. Please install pyvips to use this backend.")
    
    # print(f"Opening WSI with pyvips: {slide_path}")
    
    # Load the WSI using pyvips
    try:
        # Load the image at the specified level
        wsi_image = pyvips.Image.new_from_file(slide_path, level=args.WSI_level)
    except Exception as e:
        print(f"Error loading WSI with pyvips: {e}")
        raise
    
    original_width = wsi_image.width
    original_height = wsi_image.height
    # print(f"WSI size at level {args.WSI_level}: {original_width}x{original_height} pixels")
    
    # Create thumbnail for background mask generation
    max_thumb_size = 1000
    if max(original_height, original_width) > max_thumb_size:
        scale = max_thumb_size / max(original_height, original_width)
        thumb_width = int(original_width * scale)
        thumb_height = int(original_height * scale)
        thumbnail_vips = wsi_image.resize(scale)
    else:
        thumbnail_vips = wsi_image
        scale = 1.0
    
    # Convert pyvips image to numpy array for cv2 processing
    thumbnail_np = np.ndarray(buffer=thumbnail_vips.write_to_memory(),
                             dtype=np.uint8,
                             shape=[thumbnail_vips.height, thumbnail_vips.width, thumbnail_vips.bands])
    
    # Handle different color spaces
    if thumbnail_vips.bands == 1:
        # Grayscale - convert to RGB
        thumbnail = cv2.cvtColor(thumbnail_np, cv2.COLOR_GRAY2RGB)
    elif thumbnail_vips.bands == 3:
        # RGB format
        thumbnail = thumbnail_np
    elif thumbnail_vips.bands == 4:
        # RGBA - convert to RGB
        thumbnail = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError(f"Unsupported number of bands: {thumbnail_vips.bands}")
    
    # Process black pixels
    black_pixel = np.where((thumbnail[:, :, 0] < 50) & (thumbnail[:, :, 1] < 50) & (thumbnail[:, :, 2] < 50))
    thumbnail[black_pixel] = [255, 255, 255]
    
    # Convert RGB to BGR for cv2 operations in get_bg_mask
    thumbnail_bgr = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
    
    # Generate background mask
    bg_mask = get_bg_mask(thumbnail_bgr, kernel_size=args.kernel_size)
    
    # Calculate patch parameters
    x_size = args.patch_w
    y_size = args.patch_h
    x_overlap = args.overlap_w
    y_overlap = args.overlap_h
    
    # Generate valid patch coordinates
    coordinates = generate_patch_coordinates(original_width, original_height, x_size, y_size, x_overlap, y_overlap, bg_mask, args.blank_TH)
    
    # Update progress bar
    patch_progress.total = len(coordinates)
    patch_progress.refresh()
    
    # print(f"Starting to extract {len(coordinates)} patches using pyvips...")
    
    # Collect patch position information for annotation
    patch_positions = []
    
    # Process patches
    patch_count = 0
    for idx, (i, j) in enumerate(coordinates):
        x_start = int(i * (x_size - x_overlap))
        y_start = int(j * (y_size - y_overlap))
        
        # Extract patch using pyvips
        try:
            # Crop the patch from the WSI
            patch_vips = wsi_image.crop(x_start, y_start, x_size, y_size)
            
            # Convert to numpy array
            patch_np = np.ndarray(buffer=patch_vips.write_to_memory(),
                                 dtype=np.uint8,
                                 shape=[patch_vips.height, patch_vips.width, patch_vips.bands])
            
            # Handle different color spaces for patch
            if patch_vips.bands == 1:
                # Grayscale - convert to RGB
                patch_rgb = cv2.cvtColor(patch_np, cv2.COLOR_GRAY2RGB)
            elif patch_vips.bands == 3:
                # RGB format
                patch_rgb = patch_np
            elif patch_vips.bands == 4:
                # RGBA - convert to RGB
                patch_rgb = cv2.cvtColor(patch_np, cv2.COLOR_RGBA2RGB)
            else:
                raise ValueError(f"Unsupported number of bands in patch: {patch_vips.bands}")
            
            # Convert to PIL Image for saving
            patch_pil = Image.fromarray(patch_rgb)
            
            # Pad if necessary
            if patch_pil.size != (x_size, y_size):
                padded_image = Image.new('RGB', (x_size, y_size), (255, 255, 255))
                padded_image.paste(patch_pil, (0, 0))
                patch_pil = padded_image
            
            # Save patch
            patch_path = os.path.join(save_path, f"no{idx:06d}_{x_start:09d}x_{y_start:09d}y.jpg")
            patch_pil.save(patch_path, 'JPEG', quality=100)
            
            # Record patch position (scaled to thumbnail coordinates)
            patch_x = int(x_start * scale)
            patch_y = int(y_start * scale)
            patch_w = int(x_size * scale)
            patch_h = int(y_size * scale)
            patch_positions.append((patch_x, patch_y, patch_w, patch_h))
            
            patch_count += 1
            patch_progress.update(1)
            
        except Exception as e:
            print(f"Error processing patch {idx} at ({x_start}, {y_start}): {e}")
            continue
    
    # print(f"Extracted {patch_count} patches using pyvips, saving thumbnails...")
    
    # Save regular thumbnail (same as opensdpc format)
    thumbnail_save_path = os.path.join(save_path, 'thumbnail/x20_thumbnail.jpg')
    os.makedirs(os.path.dirname(thumbnail_save_path), exist_ok=True)
    x20_thumbnail = Image.fromarray(thumbnail)
    save_image(x20_thumbnail, thumbnail_save_path)
    
    # Draw annotated thumbnail (same as opensdpc format)
    draw_annotation_thumbnail(thumbnail_bgr, bg_mask, patch_positions, thumbnail_save_path)
    
    # print(f"Pyvips processing completed successfully!")
    return patch_count


def process_opensdpc_wsi(slide_path, save_path, args, patch_progress):
    # Process standard format WSI (using opensdpc) - DEFAULT BEHAVIOR
    try:
        slide = opensdpc.OpenSdpc(slide_path)
    except:
        slide = openslide.OpenSlide(slide_path)
    
    # Use the last level as thumbnail (safer than thumb_n offset)
    thumbnail_level = slide.level_count - 1
    thumbnail = np.array(slide.read_region((0, 0), thumbnail_level, slide.level_dimensions[thumbnail_level]).convert('RGB'))
    
    # Clean black pixels and generate background mask
    black_pixel = np.where((thumbnail[:, :, 0] < 50) & (thumbnail[:, :, 1] < 50) & (thumbnail[:, :, 2] < 50))
    thumbnail[black_pixel] = [255, 255, 255]
    bg_mask = get_bg_mask(thumbnail, kernel_size=args.kernel_size)

    # Get dimensions and downsample factor for target level
    read_level = args.WSI_level
    level_downsample = slide.level_downsamples[read_level]
    img_x, img_y = slide.level_dimensions[0]  # Level 0 dimensions
    
    # Calculate patch parameters in level 0 coordinates
    x_size_l0 = int(args.patch_w * level_downsample)
    y_size_l0 = int(args.patch_h * level_downsample)
    x_overlap_l0 = int(args.overlap_w * level_downsample)
    y_overlap_l0 = int(args.overlap_h * level_downsample)

    # Generate coordinates in level 0 coordinate system
    coordinates = generate_patch_coordinates(img_x, img_y, x_size_l0, y_size_l0, x_overlap_l0, y_overlap_l0, bg_mask, args.blank_TH)

    

    # Collect patch position information for annotation
    patch_positions = []
    
    for idx, (i, j) in enumerate(coordinates):
        # Calculate level 0 coordinates
        x_start = int(i * (x_size_l0 - x_overlap_l0))
        y_start = int(j * (y_size_l0 - y_overlap_l0))

        # Read patch at target level with target patch size
        img = slide.read_region((x_start, y_start), read_level, (args.patch_w, args.patch_h)).convert('RGB')
        save_image(img, os.path.join(save_path, f"no{idx:06d}_{x_start:09d}x_{y_start:09d}y.jpg"))
        
        # Record patch position for thumbnail annotation
        thumbnail_scale_x = thumbnail.shape[1] / img_x
        thumbnail_scale_y = thumbnail.shape[0] / img_y
        patch_x = int(x_start * thumbnail_scale_x)
        patch_y = int(y_start * thumbnail_scale_y)
        patch_w = int(x_size_l0 * thumbnail_scale_x)
        patch_h = int(y_size_l0 * thumbnail_scale_y)
        patch_positions.append((patch_x, patch_y, patch_w, patch_h))
        
        patch_progress.update(1)

    patch_progress.close()

    # Save regular thumbnail
    thumbnail_save_path = os.path.join(save_path, 'thumbnail/x20_thumbnail.jpg')
    os.makedirs(os.path.dirname(thumbnail_save_path), exist_ok=True)
    x20_thumbnail = Image.fromarray(thumbnail)
    save_image(x20_thumbnail, thumbnail_save_path)
    
    # Draw annotated thumbnail
    draw_annotation_thumbnail(thumbnail, bg_mask, patch_positions, thumbnail_save_path)


def func_patching(args, pair_list, thread_id):
    total_slides = len(pair_list) 
    
    for item, pair_path in enumerate(pair_list):
        slide_path = pair_path[0]
        save_path = pair_path[1]
        os.makedirs(save_path, exist_ok=True)

        # Check file format first
        if is_jpg_format(slide_path):   # JPG 
            # Process JPG format WSI (always use PIL-based processing for JPG)
            # Set initial progress bar with placeholder value since we'll update it after coordinate generation
            patch_progress = tqdm(total=1000, 
                                desc=f'THREAD {thread_id} Slide {item+1}/{total_slides} (JPG)', 
                                position=thread_id, 
                                ncols=90,  
                                leave=False)
            
            process_jpg_wsi(slide_path, save_path, args, patch_progress)
            patch_progress.close()
            
        elif hasattr(args, 'backend') and args.backend == 'pyvips':  # PYVIPS
            # Process WSI format using pyvips
            patch_progress = tqdm(total=1000, 
                                desc=f'THREAD {thread_id} Slide {item+1}/{total_slides} (PYVIPS)', 
                                position=thread_id, 
                                ncols=90,  
                                leave=False)
            
            process_pyvips_wsi(slide_path, save_path, args, patch_progress)
            patch_progress.close()
            
        else:  # OPENSDPC & OPENSLIDE
            patch_progress = tqdm(total=1000, 
                            desc=f'THREAD {thread_id} Slide {item+1}/{total_slides} (OPENSDPC)', 
                            position=thread_id, 
                            ncols=90,  
                            leave=False)
            process_opensdpc_wsi(slide_path, save_path, args, patch_progress)
            patch_progress.close()

