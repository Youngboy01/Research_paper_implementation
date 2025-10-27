import torch
import torch.nn.functional as F
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
        self.num_image_tokens = num_image_tokens
        
class SigLipVisionModel(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)
    def forward(self,pixel_values)-> tuple:
        #[batch,3,h,w]->[batch,num_patches,embed_dim]
        return self.vision_model(pixel_values=pixel_values)
    
class SigLipVisionTransformer(nn.Module):
    def __init__(self,config: SigLipVisionConfig):
        super().__init__()
        self.config  = config
        self.embed_dim = config.hidden_size
        
        self.vision_embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embed_dim,eps=config.layer_norm_eps)
    def forward(self, pixel_values: torch.Tensor):
        embeddings = self.vision_embeddings(pixel_values)
        #pass through encoder
        encoder_output = self.encoder(embeddings)
        normalized_output = self.layer_norm(encoder_output)
        return normalized_output
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
            stride=self.patch_size,#no overlapping
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
class SigLipEncoder(nn.Module):
    def __init__(self,config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.ln1 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.ln2 = nn.LayerNorm(self.embed_dim,eps=config.hidden_size)
    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        #store hidden_states as residual for connecting later as residual connection
        residual = hidden_states
        #layernorm
        hidden_states = self.ln1(hidden_states)
        #attention
        hidden_states = self.self_attn(hidden_states)
        #add connection
        hidden_states = residual+hidden_states
        #store again
        residual = hidden_states
        #now again layer norm
        hidden_states = self.ln2(hidden_states)
        #pass through mlp
        hidden_states = self.mlp(hidden_states)
        #add the skip connection
        hidden_states = residual+hidden_states
        return hidden_states
class SigLipMLP(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.linearlayer1 = nn.Linear(config.hidden_size,config.intermediate_size)
        self.linearlayer2 = nn.Linear(config.intermediate_size,config.hidden_size)
    def forward(self,hidden_states: torch.Tensor)->torch.Tensor:
        hidden_states = self.linearlayer1(hidden_states)
        #add nonlinearity
        hidden_states = F.gelu(hidden_states,approximate="tanh")
        hidden_states = self.linearlayer2(hidden_states)
        return hidden_states
        