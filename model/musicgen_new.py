import torch
import torch.nn as nn
from omegaconf import OmegaConf
from .lm_new import LMModel
import numpy as np
import random
from .codebooks_patterns import (
    CodebooksPatternProvider,
    DelayedPatternProvider,
    MusicLMPattern,
    ParallelPatternProvider,
    UnrolledPatternProvider,
    CoarseFirstPattern,
)

def get_codebooks_pattern_provider(n_q: int, cfg) -> CodebooksPatternProvider:
    """Instantiate a codebooks pattern provider object."""
    pattern_providers = {
        'parallel': ParallelPatternProvider,
        'delay': DelayedPatternProvider,
        'unroll': UnrolledPatternProvider,
        'coarse_first': CoarseFirstPattern,
        'musiclm': MusicLMPattern,
    }
    name = cfg.modeling
    kwargs = OmegaConf.to_container(cfg.get(name), resolve=True) if hasattr(cfg, name) else {}
    klass = pattern_providers[name]
    return klass(n_q, **kwargs)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

        
class MusicGenerator(nn.Module):
    def __init__(self, model_config):
        super(MusicGenerator, self).__init__()
        self.music_channels = model_config['features']['pitch_channels'] + model_config['features']['chroma_channels'] + model_config['features']['loudness_channels'] + model_config['features']['spectral_channels']
        self.conv_pitch = nn.ConvTranspose1d(in_channels=model_config['transformer']['encoder_hidden'],
                                    out_channels=model_config['features']['pitch_channels'],
                                    kernel_size=1,
                                    padding=4,
                                    stride=2)
        self.conv_chroma = nn.ConvTranspose1d(in_channels=model_config['transformer']['encoder_hidden'],
                                    out_channels=model_config['features']['chroma_channels'],
                                    kernel_size=1,
                                    padding=4,
                                    stride=2)
        self.conv_loudness = nn.ConvTranspose1d(in_channels=model_config['transformer']['encoder_hidden'],
                                    out_channels=model_config['features']['loudness_channels'],
                                    kernel_size=1,
                                    padding=4,
                                    stride=2)
        self.conv_spectral = nn.ConvTranspose1d(in_channels=model_config['transformer']['encoder_hidden'],
                                    out_channels=model_config['features']['spectral_channels'],
                                    kernel_size=1,
                                    padding=4,
                                    stride=2)
        self.conv = nn.Conv1d(in_channels=self.music_channels,
                              out_channels=model_config['transformer_lm']['dim'],
                              kernel_size=2,
                              stride=2)
        
        self.conv.apply(init_weights)
        self.conv_chroma.apply(init_weights)
        self.conv_loudness.apply(init_weights)
        self.conv_pitch.apply(init_weights)
        self.conv_spectral.apply(init_weights)
        
        n_q = model_config['transformer_lm']['n_q']
        codebooks_pattern_cfg = getattr(OmegaConf.create(model_config), 'codebooks_pattern')
        pattern_provider = get_codebooks_pattern_provider(n_q, codebooks_pattern_cfg)
        self.decoder = LMModel(pattern_provider, video_hidden = model_config['transformer']['encoder_hidden'], **model_config['transformer_lm'])
        self.dropout = nn.Dropout(model_config['transformer']['dropout'])
        
    
    def forward(self, x, target):
        pitch = self.conv_pitch(x)
        chroma = self.conv_chroma(x)
        loudness = self.conv_loudness(x)
        spectral = self.conv_spectral(x)
        if random.random() < 0.1:
            random_numbers = np.random.dirichlet(alpha=[1, 1, 1, 1])
            # random_numbers = [random.random() for _ in range(4)]
            pitch = pitch * (random_numbers[0] * 4)#random_numbers[0]#
            chroma = chroma * (random_numbers[1] * 4)#random_numbers[1]#
            loudness = loudness * (random_numbers[2] * 4)#random_numbers[2]#
            spectral = spectral * (random_numbers[3] * 4)#random_numbers[3]#
        music_in = torch.cat([pitch, chroma, loudness, spectral], dim=1)
        music_out = self.dropout(music_in)
        music_out = self.conv(music_out)
        music_out = music_out.permute(0, 2, 1)
        
        x = x.permute(0,2,1)
        output, gt_loss, pre_loss = self.decoder.compute_predictions(target, music_out, x)
        return pitch, chroma, loudness.squeeze(), spectral.squeeze(), output, gt_loss, pre_loss
    
    def inference(self, x, music_control=('-1', 1.0)):
        pitch = self.conv_pitch(x)
        chroma = self.conv_chroma(x)
        loudness = self.conv_loudness(x)
        spectral = self.conv_spectral(x)
        
        feature, weight = music_control
        if feature == 'pitch':
            pitch = pitch * weight
        elif feature == 'chroma':
            chroma = chroma * weight
        elif feature == 'loudness':
            loudness = loudness * weight
        elif feature == 'spectral':
            spectral = spectral * weight
                
        music_in = torch.cat([pitch, chroma, loudness, spectral], dim=1)
        if feature == '-1':
            music_in = music_in * weight
        music_out = music_in
        music_out = self.conv(music_out)
        music_out = music_out.permute(0, 2, 1)
        output = self.decoder.generate(conditions=music_out, max_gen_len=music_out.shape[1] * 50)
        return pitch, chroma, loudness.squeeze(), spectral.squeeze(), output

        