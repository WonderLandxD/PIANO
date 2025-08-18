
# you need to install the MIL-Lab first
# git clone https://github.com/mahmoodlab/MIL-Lab.git


from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn


class FEATHER_UNIV1_Model(nn.Module):
    def __init__(self):
        super().__init__()
        from src.builder import create_model
        self.slide_model = create_model('abmil.base.uni.pc108-24k', from_pretrained=True, num_classes=0)
    
    def forward(self, input):
        patch_features = input['feats']
        assert len(patch_features.shape) == 3

        _, log_dict = self.slide_model(patch_features, 
                               loss_fn=None, 
                               label=None, 
                               return_attention=False,
                               return_slide_feats=True
        )
        
        slide_feats = log_dict['slide_feats']
        return slide_feats
        

if __name__ == "__main__":
    model = create_model('abmil.base.uni.pc108-24k', from_pretrained=True, num_classes=0)

    features = torch.randn(1, 100 ,1024)
    _, log_dict = model(features, 
                               loss_fn=None, 
                               label=None, 
                               return_attention=False,
                               return_slide_feats=True
    )
    
    slide_feats = log_dict['slide_feats']
    print(slide_feats.shape)
    


    # load the model from the huggingface
    # model = AutoModel.from_pretrained("MahmoodLab/abmil.base.uni.pc108-24k", trust_remote_code=True)
    # print(model)
