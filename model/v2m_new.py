import torch
import torch.nn as nn
from .video import FeatureTransformer
from .musicgen_new import MusicGenerator

class V2M(nn.Module):
    def __init__(self, model_config):
        super(V2M, self).__init__()
        self.video_transformer = FeatureTransformer(model_config)
        self.music_generator = MusicGenerator(model_config)
        self.initialize_parameters(model_config)

    def initialize_parameters(self, model_config):
            
        proj_std = (model_config['transformer']['encoder_hidden'] ** -0.5) * ((2 * model_config['transformer']['layers']) ** -0.5)
        attn_std = model_config['transformer']['encoder_hidden'] ** -0.5
        fc_std = (2 * model_config['transformer']['encoder_hidden']) ** -0.5
        for block in self.video_transformer.time_attn.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)


    def forward(self, semantic, color, position, move, area, mel):
        video_out = self.video_transformer(semantic, color, position, move, area)
    
        pitch, chroma, loudness, spectral, music_out, gt_loss, pre_loss = self.music_generator(video_out, mel)
        return pitch, chroma, loudness, spectral, music_out, gt_loss, pre_loss
    
    def infer(self, semantic, color, position, move, area, music_control=('-1', 1.0)):
        video_out = self.video_transformer.inference(semantic, color, position, move, area)
        pitch, chroma, loudness, spectral, music_out = self.music_generator.inference(video_out, music_control)
        return pitch, chroma, loudness, spectral, music_out