import torch
import torch.nn as nn
class SigLipVisionConfig:
    """Referred from https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/configuration_siglip.py"""

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.num_image_tokens = num_image_tokens
class SigLipVisionEmbeddings(nn.Module):
    def __init__(self,config:SigLipVisionConfig):
        super().__init__()
        self.config= config
        self.embed_dim=config.hidden_size
        self.image_size=config.image_size
        self.patch_size=config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid'
        )
        self.num_patches = (self.image_size//self.patch_size)**2
        self.num_posn = self.num_patches
        self.positional_embedding = nn.Embedding(self.num_posn,self.embed_dim)
        self.register_buffer(
            'position_ids',
            torch.arange(self.num_posn).expand((-1,1)),
            persistent=False
        )
        
    def forward(self,pixel_values:torch.FloatTensor)->torch.Tensor:
        _,_,height,width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeds = patch_embeds.flatten(2)
        embeds = embeds.transpose(1,2)
        embeds = embeds + self.positional_embedding(self.position_ids)
        return embeds