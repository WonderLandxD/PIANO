import torch
from .HIPT_4K.hipt_4k import HIPT_4K
from huggingface_hub import hf_hub_download

class HIPTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        repo_id = 'JWonderLand/HIPT_unofficial' 
        vit256_path = hf_hub_download(repo_id=repo_id, filename="vit256_small_dino.pth")
        vit4k_path = hf_hub_download(repo_id=repo_id, filename="vit4k_xs_dino.pth")
        
        self.backbone = HIPT_4K(
            model256_path=vit256_path, 
            model4k_path=vit4k_path, 
            device256=torch.device('cpu'), 
            device4k=torch.device('cpu')
        )
        

        self.image_preprocess = None
        self.output_dim = 384
	
    def forward(self, input):
        patch_features = input['feats']

        # Calculate target length for padding
        N = patch_features.shape[1]
        target_N = ((N + 255) // 256) * 256

        # If padding is needed
        if N < target_N:
            # Calculate padding size
            padding_size = target_N - N
            # Use original patch features for padding
            # Create padding tensor
            padding_list = []
            remaining = padding_size
            
            # Loop until padding space is filled
            while remaining > 0:
                # Get current available patch count (not exceeding remaining padding size)
                current_fill = min(remaining, N)
                # Add to padding list
                padding_list.append(patch_features[:, :current_fill, :])
                # Update remaining padding size
                remaining -= current_fill
            
            # Concatenate all padding parts
            padding = torch.cat(padding_list, dim=1) if len(padding_list) > 1 else padding_list[0]
            # Concatenate along dimension 1 (patch dimension)
            patch_features = torch.cat([patch_features, padding], dim=1)

        _, new_N, C = patch_features.shape
        mini_B = int(new_N / 256)
        features_cls256 = patch_features.reshape(mini_B, 256, C).reshape(mini_B, C, 16, 16)

        features_cls4k = self.backbone.model4k.forward(features_cls256)
        final_features = features_cls4k.mean(dim=0, keepdim=True)
        return final_features


if __name__ == '__main__':
  input = {
    'feats': torch.randn(1, 10000, 384).cuda()
  }
  model = HIPTModel().cuda()
  output = model(input)
  print(output.shape)
  