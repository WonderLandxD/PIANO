'''
Based on HIPT: https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/hipt_4k.py
'''

### Dependencies
# Base Dependencies
import os
import pickle
import sys

# LinAlg / Stats / Plotting Dependencies
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.multiprocessing
from torchvision import transforms
from einops import rearrange
torch.multiprocessing.set_sharing_strategy('file_system')

# Local Dependencies
from .hipt_model_utils import get_vit256, get_vit4k


class HIPT_4K(torch.nn.Module):
	"""
	HIPT Model (ViT-4K) for encoding non-square images (with [256 x 256] patch tokens), with 
	[256 x 256] patch tokens encoded via ViT-256 using [16 x 16] patch tokens.
	"""
	def __init__(self, 
		model256_path: str = '../Checkpoints/vit256_small_dino.pth',
		model4k_path: str = '../Checkpoints/vit4k_xs_dino.pth', 
		device256=torch.device('cuda:0'), 
		device4k=torch.device('cuda:1')):

		super().__init__()
		self.model256 = get_vit256(pretrained_weights=model256_path).to(device256)
		self.model4k = get_vit4k(pretrained_weights=model4k_path).to(device4k)
		self.device256 = device256
		self.device4k = device4k
	
	def forward(self, x):
		"""
		Forward pass of HIPT (given an image tensor x), outputting the [CLS] token from ViT-4K.
		1. x is center-cropped such that the W / H is divisible by the patch token size in ViT-4K (e.g. - 256 x 256).
		2. x then gets unfolded into a "batch" of [256 x 256] images.
		3. A pretrained ViT-256 model extracts the CLS token from each [256 x 256] image in the batch.
		4. These batch-of-features are then reshaped into a 2D feature grid (of width "w_256" and height "h_256".)
		5. This feature grid is then used as the input to ViT-4K, outputting [CLS]_4K.
		
		Args:
			- x (torch.Tensor): [1 x C x W' x H'] image tensor.
		
		Return:
			- features_cls4k (torch.Tensor): [1 x 192] cls token (d_4k = 192 by default).
		"""
		batch_256, w_256, h_256 = self.prepare_img_tensor(x)                    # 1. [1 x 3 x W x H] 
		batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)           # 2. [1 x 3 x w_256 x h_256 x 256 x 256] 
		batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)
										  
		features_cls256 = []
		for mini_bs in range(0, batch_256.shape[0], 256):                       # 3. B may be too large for ViT-256. We further take minibatches of 256.
			minibatch_256 = batch_256[mini_bs:mini_bs+256].to(self.device256, non_blocking=True)
			features_cls256.append(self.model256(minibatch_256).detach().cpu()) # 3. Extracting ViT-256 features from [256 x 3 x 256 x 256] image batches.

		features_cls256 = torch.vstack(features_cls256)                         # 3. [B x 384], where 384 == dim of ViT-256 [ClS] token.
		features_cls256 = features_cls256.reshape(w_256, h_256, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0) 
		features_cls256 = features_cls256.to(self.device4k, non_blocking=True)  # 4. [1 x 384 x w_256 x h_256]
		features_cls4k = self.model4k.forward(features_cls256)                  # 5. [1 x 192], where 192 == dim of ViT-4K [ClS] token.
		return features_cls4k
	
	
	def forward_asset_dict(self, x: torch.Tensor):
		"""
		Forward pass of HIPT (given an image tensor x), with certain intermediate representations saved in 
		a dictionary (that is to be stored in a H5 file). See walkthrough of how the model works above.
		
		Args:
			- x (torch.Tensor): [1 x C x W' x H'] image tensor.
		
		Return:
			- asset_dict (dict): Dictionary of intermediate feature representations of HIPT and other metadata.
				- features_cls256 (np.array): [B x 384] extracted ViT-256 cls tokens
				- features_mean256 (np.array): [1 x 384] mean ViT-256 cls token (exluding non-tissue patches)
				- features_4k (np.array): [1 x 192] extracted ViT-4K cls token.
				- features_4k (np.array): [1 x 576] feature vector (concatenating mean ViT-256 + ViT-4K cls tokens)
	
		"""
		batch_256, w_256, h_256 = self.prepare_img_tensor(x)
		batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)
		batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')
		
		features_cls256 = []
		for mini_bs in range(0, batch_256.shape[0], 256):
			minibatch_256 = batch_256[mini_bs:mini_bs+256].to(self.device256, non_blocking=True)
			features_cls256.append(self.model256(minibatch_256).detach().cpu())

		features_cls256 = torch.vstack(features_cls256)
		features_mean256 = features_cls256.mean(dim=0).unsqueeze(dim=0)

		features_grid256 = features_cls256.reshape(w_256, h_256, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0)
		features_grid256 = features_grid256.to(self.device4k, non_blocking=True)
		features_cls4k = self.model4k.forward(features_grid256).detach().cpu()
		features_mean256_cls4k = torch.cat([features_mean256, features_cls4k], dim=1)
		
		asset_dict = {
			'features_cls256': features_cls256.numpy(),
			'features_mean256': features_mean256.numpy(),
			'features_cls4k': features_cls4k.numpy(),
			'features_mean256_cls4k': features_mean256_cls4k.numpy()
		}
		return asset_dict
			
	def prepare_img_tensor(self, img: torch.Tensor, patch_size=256):
		"""
		Helper function that takes a non-square image tensor, and takes a center crop s.t. the width / height
		are divisible by 256.
		
		(Note: "_256" for w / h is should technically be renamed as "_ps", but may not be easier to read.
		Until I need to make HIPT with patch_sizes != 256, keeping the naming convention as-is.)
		
		Args:
			- img (torch.Tensor): [1 x C x W' x H'] image tensor.
			- patch_size (int): Desired patch size to evenly subdivide the image.
		
		Return:
			- img_new (torch.Tensor): [1 x C x W x H] image tensor, where W and H are divisble by patch_size.
			- w_256 (int): # of [256 x 256] patches of img_new's width (e.g. - W/256)
			- h_256 (int): # of [256 x 256] patches of img_new's height (e.g. - H/256)
		"""
		make_divisble = lambda l, patch_size: (l - (l % patch_size))
		b, c, w, h = img.shape
		load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
		w_256, h_256 = w // patch_size, h // patch_size
		img_new = transforms.CenterCrop(load_size)(img)
		return img_new, w_256, h_256
	


if __name__ == "__main__":
	from huggingface_hub import hf_hub_download
	vit256_path = hf_hub_download(repo_id="JWonderLand/HIPT_unofficial", filename="vit256_small_dino.pth")
	vit4k_path = hf_hub_download(repo_id="JWonderLand/HIPT_unofficial", filename="vit4k_xs_dino.pth")
	# vit256_path = "/mnt/sdb/ljw/piano_weights/HIPT/HIPT_unofficial/vit256_small_dino.pth"
	# vit4k_path = "/mnt/sdb/ljw/piano_weights/HIPT/HIPT_unofficial/vit4k_xs_dino.pth"
	model = HIPT_4K(
		model256_path=vit256_path, 
		model4k_path=vit4k_path, 
		device256=torch.device('cpu'), 
		device4k=torch.device('cpu'))
	img_tensor = torch.randn(1, 3, 224, 224)
	# features = model.forward(img_tensor)
	features = model.model256(img_tensor)
	print(features.shape)