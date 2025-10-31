import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional,Tuple
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
class SigLipAttention(nn.Module):#will not contain causal masking as vision doesn't need to be autoregressive
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.embed_dim=config.hidden_size
        self.head_dim = self.embed_dim//self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout#though we wont be using it
        
        self.W_q = nn.Linear(self.embed_dim,self.embed_dim)
        self.W_v = nn.Linear(self.embed_dim,self.embed_dim)
        self.W_k = nn.Linear(self.embed_dim,self.embed_dim)
        self.W_o = nn.Linear(self.embed_dim,self.embed_dim)
    def forward(self,hidden_states: torch.Tensor)->Tuple[torch.Tensor,Optional[torch.Tensor]]:
        #hidden_states = [batch_size,num_patches,embed_dim]
        batch_size,seq_len,_ = hidden_states.size()
        #key = [batch_size,num_patches,embed_dim]
        key = self.W_k(hidden_states)
        #query = [batch_size,num_patches,embed_dim]
        query = self.W_q(hidden_states)
        #value = [batch_size,num_patches,embed_dim]
        value = self.W_v(hidden_states)
        #embed_dim = head_dim*num_heads
        #query/key/value = [batch_size,num_patches,num_heads,head_dim]->[batch_size,num_heads,num_patches,head_dim]
        query = query.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        key = key.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        value = value.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        #now calculate attention weights using the formula. attention_weights = [batch_size,num_heads,num_patches,num_patches]
        attn_weights = (torch.matmul(query,key.transpose(2,3))*self.scale)
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        #applying softmax rowise
        attn_weights = F.softmax(attn_weights,dim=-1,dtype=torch.float32).to(query.dtype)
        #dropout exists but is never used in the params of paligemma so not writing it
        #now multiply attn_weight with the value sequence
        attn_output = torch.matmul(attn_weights,value)
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        #[Batch_size,num_heads,num_patches,head_dim]->[Batch_size,num_patches,num_heads,head_dim]
        attn_output = attn_output.transpose(1,2).contiguous()
        #multpily back the last two dims [Batch_size,num_patches,num_heads,head_dim]->[Batch_size,num_patches,embed_dim]
        attn_output = attn_output.reshape(batch_size,seq_len,self.embed_dim)
        attn_output = self.W_o(attn_output)
        return attn_output,attn_weights
    