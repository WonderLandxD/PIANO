import argparse
import os
import csv
import logging
import importlib.util
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _load_func_patching():
    """Load slide_patch_tools from this repo instead of an installed `piano`.

    Loading the file directly also avoids importing `piano/__init__.py`, which
    pulls in the heavy model dependencies that patching does not need.
    """
    piano_root = Path(__file__).resolve().parents[2]
    tools_path = piano_root / 'piano' / 'wsi_preprocess' / 'slide_patch_tools.py'
    if not tools_path.is_file():
        raise FileNotFoundError(f'Cannot locate slide_patch_tools.py at: {tools_path}')

    spec = importlib.util.spec_from_file_location('piano_slide_patch_tools', tools_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.func_patching


func_patching = _load_func_patching()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('patching.log'),
            logging.StreamHandler()
        ]
    )

def read_slide_list(csv_file_path):
    slide_list = []
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in tqdm(reader):
            if not row or not row[0].strip():
                continue
            slide_list.append(row[0].strip())
    return slide_list

def infer_data_root(slide_list):
    """Deepest directory shared by every slide in the list."""
    if not slide_list:
        return None
    return Path(os.path.commonpath([str(Path(p).parent) for p in slide_list]))

def resolve_save_path(slide_path, save_dir, data_root):
    """Mirror the source tree (project / stage / case) under save_dir."""
    slide_path = Path(slide_path)
    relative_dir = None
    if data_root is not None:
        try:
            relative_dir = slide_path.parent.relative_to(data_root)
        except ValueError:
            logging.warning(f'{slide_path} is outside data_root {data_root}, flattening its output path')
    if relative_dir is None:
        relative_dir = Path('')
    return Path(save_dir) / relative_dir / slide_path.stem

def generate_pair_list(slide_list, save_dir, data_root):
    pair_list = []
    for slide_path in tqdm(slide_list):
        save_path = resolve_save_path(slide_path, save_dir, data_root)
        thumb_path = save_path / 'thumbnail' / 'x20_thumbnail.jpg'
        if not thumb_path.exists():
            pair_list.append([slide_path, str(save_path)])
    return pair_list

def distribute_processing_wsi2patch(pair_list, num_thread, args):
    sub_pair_list = np.array_split(pair_list, num_thread)

    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        futures = {executor.submit(func_patching, args, sublist, i): i for i, sublist in enumerate(sub_pair_list)}

        for future in as_completed(futures):
            thread_index = futures[future]
            try:
                result = future.result()
                # logging.info(result)  
            except Exception as exc:
                logging.error(f'Thread {thread_index} generated an exception: {exc}', exc_info=True)

def validate_args(args):
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    if not 1 <= args.jpg_quality <= 100:
        raise ValueError("jpg_quality must be between 1 and 100")
    if args.mpp_tolerance <= 0:
        raise ValueError("mpp_tolerance must be greater than 0")
    if args.overlap_w >= args.patch_w or args.overlap_h >= args.patch_h:
        raise ValueError("Overlap must be smaller than patch size")
    if args.n_thread <= 0:
        raise ValueError("Number of threads must be greater than 0")
    if args.data_root is not None and not os.path.isdir(args.data_root):
        raise NotADirectoryError(f"data_root not found: {args.data_root}")

def parse():
    parser = argparse.ArgumentParser(description='Multithreaded patch generation for histopathology whole slide images using PIANO and opensdpc library.')
    parser.add_argument('--n_thread', type=int, default=16, help='Number of threads to use')
    # Hyperparameters for patch generation
    parser.add_argument('--patch_w', type=int, default=256, help='Width of patch')
    parser.add_argument('--patch_h', type=int, default=256, help='Height of patch')     
    parser.add_argument('--overlap_w', type=int, default=0, help='Overlap width of patch')
    parser.add_argument('--overlap_h', type=int, default=0, help='Overlap height of patch')             
    parser.add_argument('--magnification', type=str, required=True, choices=['40x', '20x'],
                        help='Target magnification; the pyramid level is picked by mpp and slides without a '
                             'matching level are skipped')
    parser.add_argument('--mpp_tolerance', type=float, default=0.15,
                        help='Relative mpp tolerance when matching --magnification')
    parser.add_argument('--allow_downscale', action='store_true',
                        help='When no level matches --magnification, read a larger region from a finer '
                             'level and shrink it to the target resolution instead of skipping the slide')
    parser.add_argument('--blank_TH', type=float, default=0.7, help='Blank threshold for patch cutting')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for mask operations')
    parser.add_argument('--thumb_n', type=float, default=1, help='Thumbnail layer index')
    parser.add_argument('--jpg_quality', type=int, default=40, help='JPEG quality of saved patches (1-100)')
    
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save patches')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root of the source WSI tree; the sub-tree below it (project/stage/case) is mirrored '
                             'under save_dir. Inferred from the CSV paths when omitted.')
    parser.add_argument('--append_csv_name', action='store_true',
                        help='Insert an extra directory named after the CSV file below save_dir')
    return parser.parse_args()

if __name__ == '__main__':
    setup_logging()
    args = parse()
    validate_args(args)

    csv_file_path = Path(args.csv_path)
    save_dir = Path(args.save_dir)
    if args.append_csv_name:
        save_dir = save_dir / csv_file_path.stem

    slide_list = read_slide_list(csv_file_path)

    data_root = Path(args.data_root) if args.data_root else infer_data_root(slide_list)
    logging.info(f'Data root: {data_root}')
    logging.info(f'Save dir: {save_dir}')
    logging.info(f'Level selection: by mpp, target {args.magnification} '
                 f'(tolerance {args.mpp_tolerance:.0%}, downscale fallback {"on" if args.allow_downscale else "off"})')
    logging.info(f'Patch: {args.patch_w}x{args.patch_h}, JPEG quality {args.jpg_quality}')

    pair_list = generate_pair_list(slide_list, save_dir, data_root)

    logging.info(f'All data number: {len(slide_list)}, unprocessed data number: {len(pair_list)}')
    if pair_list:
        logging.info(f'Example output dir: {pair_list[0][1]}')
    distribute_processing_wsi2patch(pair_list, args.n_thread, args)
