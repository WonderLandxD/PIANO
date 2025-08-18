import torch
import torch.nn as nn
from .raw_protea import ProTEA

class PROTEAModel(nn.Module):
    def __init__(self):
        super().__init__()

        model_pt_path = '/mnt/sdb/ljw/ProTEA/checkpoints_new/protea_best.pth'
        llm_decoder_ckpt_path = '/mnt/sdb/ljw/SqrayNet/RawModels/qwen/Qwen2.5-0.5B-Instruct'

        self.slide_model = ProTEA(input_dim=1024, num_clusters=4, num_experts=4, llm_decoder_ckpt_path=llm_decoder_ckpt_path, loss_type='clip')
        model_ckpt = torch.load(model_pt_path, map_location='cpu', weights_only=False)
        self.slide_model.load_state_dict(model_ckpt['model'], strict=True)

    def forward(self, input):
        patch_features = input['feats']
        assert len(patch_features.shape) == 3

        output_dict = self.slide_model(patch_features, input_ids=None, train_mode=False)
        output = output_dict['image_embeds']
        return output


if __name__ == '__main__':
    model = PROTEAModel()
    input = {'feats': torch.randn(1, 1024, 1024)}
    output = model(input)
    print(output.shape)