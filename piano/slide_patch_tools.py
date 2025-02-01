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

def save_image(img, path):
    img.save(path)
    


def func_patching(args, pair_list, thread_id):
    total_slides = len(pair_list) 
    
    for item, pair_path in enumerate(pair_list):
        slide_path = pair_path[0]
        save_path = pair_path[1]
        os.makedirs(save_path, exist_ok=True)

        slide = opensdpc.OpenSdpc(slide_path)
        thumbnail_level = slide.level_count - args.thumb_n
        thumbnail = np.array(slide.read_region((0, 0), thumbnail_level, slide.level_dimensions[thumbnail_level]).convert('RGB'))
        
        black_pixel = np.where((thumbnail[:, :, 0] < 50) & (thumbnail[:, :, 1] < 50) & (thumbnail[:, :, 2] < 50))
        thumbnail[black_pixel] = [255, 255, 255]
        bg_mask = get_bg_mask(thumbnail, kernel_size=args.kernel_size)

        zoom_scale = round(slide.level_downsamples[0]) / round(slide.level_downsamples[1])
        x_size = int(args.patch_w / zoom_scale)
        y_size = int(args.patch_h / zoom_scale)
        x_overlap = int(args.overlap_w / zoom_scale)
        y_overlap = int(args.overlap_h / zoom_scale)
        img_x, img_y = slide.level_dimensions[0]

        coordinates = generate_patch_coordinates(img_x, img_y, x_size, y_size, x_overlap, y_overlap, bg_mask, args.blank_TH)

        patch_progress = tqdm(total=len(coordinates), 
                            desc=f'THREAD {thread_id} Slide {item+1}/{total_slides}', 
                            position=thread_id, 
                            ncols=90,  
                            leave=False)

        for idx, (i, j) in enumerate(coordinates):
            x_start = int(i * (x_size - x_overlap))
            y_start = int(j * (y_size - y_overlap))
            x_offset = int(x_size / pow(1/zoom_scale, args.WSI_level))
            y_offset = int(y_size / pow(1/zoom_scale, args.WSI_level))

            img = slide.read_region((x_start, y_start), args.WSI_level, (x_offset, y_offset)).convert('RGB')
            save_image(img, os.path.join(save_path, f"no{idx:06d}_{x_start:09d}x_{y_start:09d}y.jpg"))
            patch_progress.update(1)

        patch_progress.close()

        thumbnail_save_path = os.path.join(save_path, 'thumbnail/x20_thumbnail.jpg')
        os.makedirs(os.path.dirname(thumbnail_save_path), exist_ok=True)
        x20_thumbnail = Image.fromarray(thumbnail)
        save_image(x20_thumbnail, thumbnail_save_path)
