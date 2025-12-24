import torch
import torch.nn as nn
import math
from collections import OrderedDict

def ShotEncoding(semantic): # [B * shot * object * 768]
    semantic = semantic.permute(0,2,1,3)
    batch = semantic.shape[0]
    object = semantic.shape[1]
    max_len = semantic.shape[2]
    d_model = semantic.shape[-1]
    se = torch.zeros(batch, object, max_len,  d_model).to(semantic.device)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                            -(math.log(10000.0) / d_model))
    se[:, :, :, 0::2] = torch.sin(position * div_term)
    se[:, :, :, 1::2] = torch.cos(position * div_term)
    semantic = semantic + se
    semantic = semantic.reshape(batch, -1, d_model)
    return semantic
    
def ColorEncoding(color): # [B * objects * 3 * 256]
    ce = torch.zeros_like(color).to(color.device)
    d_model = color.shape[-1]
    position = torch.arange(0, d_model).reshape(1,1,1,d_model).to(color.device)
    color_new = position * color
    div_term = torch.exp(torch.arange(0, d_model) *
                            -(math.log(10000.0) / d_model)).to(color.device)
    ce[:,:,:,:] = torch.sin(color_new * div_term)
    ce = ce.reshape(color.shape[0], color.shape[1], -1)
    return ce

def PositionEncoding(position, semantic):
    pe = torch.zeros_like(semantic).to(semantic.device)
    d_model = int(semantic.shape[-1] / 2)
    position_x = position[:,:,0].unsqueeze(2)
    position_y = position[:,:,1].unsqueeze(2)
    div_term = (torch.arange(0, d_model, 2) / d_model).to(semantic.device)
    pe[:, :, 0:d_model:2] = torch.sin(position_x) * div_term
    pe[:, :, 1:d_model:2] = torch.cos(position_x) * div_term
    pe[:, :, d_model::2] = torch.sin(position_y) * div_term
    pe[:, :, d_model+1::2] = torch.cos(position_y) * div_term
    semantic = semantic + pe
    return semantic
    
def RandomDrop(in_tensor, drop_rate=0.15):
    if in_tensor.dim() == 4:
        tensor = in_tensor.reshape(in_tensor.shape[0], in_tensor.shape[1], -1)
    else:
        tensor = in_tensor
    all_zero_mask = (tensor == 0).all(dim=-1)

    not_all_zero_mask = ~all_zero_mask
    not_all_zero_indices = torch.nonzero(not_all_zero_mask, as_tuple=False)

    num_to_zero = int(torch.ceil(torch.tensor(len(not_all_zero_indices) * drop_rate)).item())
    if num_to_zero > 0:
        chosen_indices = not_all_zero_indices[torch.randperm(len(not_all_zero_indices))[:num_to_zero]]
        if in_tensor.dim() == 4:
            for idx in chosen_indices:
                b, s= idx
                in_tensor[b, s, :, :] = 0
        elif in_tensor.dim() == 3:
            for idx in chosen_indices:
                b, s = idx
                in_tensor[b, s, :] = 0
    return in_tensor

def BlockMM(x, y, block1=(2, 2), block2=(1, 2)):

    assert block1[0] == block1[1]
    assert block1[1] == block2[1]
    assert x.shape[-3] == y.shape[-2]

    # Reshape x and y according to blocks for easier batch matrix multiplication
    width = block2[1]
    batch_size, time, _, _,_ = x.shape
    _, y_height, y_width = y.shape
    
    # Reshape x to (batch_size, num_blocks_x, block_height, block_width)
    # x = x.view(batch_size, x_height // block1[0], block1[0], x_width // block1[1], block1[1])
    # x = x.permute(0, 1, 3, 2, 4)#.contiguous().view(batch_size, x_height // block1[0], x_width // block1[1], block1[0], block1[1])

    # Reshape y to (batch_size, block_height_y, num_blocks_y, block_width_y)
    y = y.view(batch_size, y_height, y_width // width, width)
    y = y.permute(0, 2, 1, 3)
    
    # Use batch matrix multiplication
    result = torch.einsum('bxijk,byik->bxyj', x, y)

    # Reshape to desired output shape
    result = result.reshape(batch_size, time, y_width)
    
    return result

def PositionMoving(semantic, move, block1=(2, 2), block2=(1, 2)):
    x = move[:,:,:,0]
    y = move[:,:,:,1]
    
    batch, time, object = x.shape[0], x.shape[1], x.shape[2]
    hidden = int(semantic.shape[-1] / 2)
    cos_x = torch.cos(x)
    sin_x = torch.sin(x)
    cos_y = torch.cos(y)
    sin_y = torch.sin(y)

    x_transform = torch.empty(batch, time, object, block1[0], block1[1]).to(semantic.device)
    y_transform = torch.empty(batch, time, object, block1[0], block1[1]).to(semantic.device)

    x_transform[:, :, :, 0, 0] = cos_x
    x_transform[:, :, :, 0, 1] = sin_x
    x_transform[:, :, :, 1, 0] = -sin_x
    x_transform[:, :, :, 1, 1] = cos_x
    y_transform[:, :, :, 0, 0] = cos_y
    y_transform[:, :, :, 0, 1] = sin_y
    y_transform[:, :, :, 1, 0] = -sin_y
    y_transform[:, :, :, 1, 1] = cos_y
    
    x_move = BlockMM(x_transform, semantic[:,:,:hidden], block1, block2)
    y_move = BlockMM(y_transform, semantic[:,:,hidden:], block1, block2)
    
    return torch.cat([x_move, y_move], dim=-1)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class FeatureTransformer(nn.Module):
    def __init__(self, model_config):
        super(FeatureTransformer, self).__init__()
        self.proj = nn.Linear(768, model_config['transformer']['encoder_hidden']) #768=CLIP_hidden=3*256(rgb)
        self.time_attn = Transformer(model_config['transformer']['encoder_hidden'], model_config['transformer']['layers'], 4)
        self.dropout = nn.Dropout(model_config['transformer']['dropout'])
    
    def forward(self, semantic, color, position, move, area):
    
        semantic = RandomDrop(semantic)
        area = RandomDrop(area)
        position = RandomDrop(position)
        color = RandomDrop(color)
        move = RandomDrop(move)
    
        semantic = ShotEncoding(semantic)
        semantic = area * semantic
        position = PositionEncoding(position, semantic)
        color = ColorEncoding(color)
        semantic = semantic + position + color
        semantic = semantic.to(torch.float32)

        video = PositionMoving(semantic, move)
        video = self.proj(video)
        visual_feature = self.time_attn(video)
        visual_feature = visual_feature.permute(0, 2, 1)
        visual_feature = self.dropout(visual_feature)
        return visual_feature
    
    def inference(self, semantic, color, position, move, area):
        semantic = ShotEncoding(semantic)
        semantic = area * semantic
        position = PositionEncoding(position, semantic)
        color = ColorEncoding(color)
        semantic = semantic + color + position #
        semantic = semantic.to(torch.float32)

        video = PositionMoving(semantic, move)
        video = self.proj(video)
        visual_feature = self.time_attn(video)
        visual_feature = visual_feature.permute(0, 2, 1)
        visual_feature = self.dropout(visual_feature)
        return visual_feature
        
       
            
        