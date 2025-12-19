# this is our language model
import torch
import torch.nn as nn
import torch.nn.functional as F
from sigLip import SiglipVisionConfig, SiglipVisionModel
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math


class KVCache:
    def __init__(self) -> None:
        self.key_cache = List[torch.Tensor] = []
        self.value_cache = List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        # the shape of key cache is [batch_size, num_heads, seq_len, head_dim]
        return self.key_cache[0].shape[-2]

    def update(
        self, key_state: torch.Tensor, value_state: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # if layer idx is greater than length of key cache, we need to append the new key and value states
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_state)
            self.value_cache.append(value_state)
        else:
            # concatenate the new key and value states to the existing ones
            # shape : [batch_size, num_heads_key_value, seq_len, head_dim]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_state], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_state], dim=-2
            )
        # return the updated key and value caches
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def repeat_kv(hidden_states: torch.Tensor, num_repeat: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if num_repeat == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, num_repeat, seq_len, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * num_repeat, seq_len, head_dim
    )


class GemmaConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,  # wont be used in inferencing
        image_token_idx=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_idx
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False  # hf uses we wont
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim


class PaliGemmaMultimodalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        projected_features = self.linear(image_features)
        return projected_features


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def norm(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x

    def forward(self, x):
        output = self.norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fcgate = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )  # used by the activation function
        self.fcup = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.fcdown = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        x = self.fcdown(F.gelu(self.fcgate(x), approximate="tanh") * self.fcup(x))


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, deivce=None):
        super().__init__()
        self.dim = dim  # it should be equal to head dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = deivce
        # formula : theta_i = base^(2i/dim)
        # inv freq = 1/theta_i = 1/(base^(2i/dim))
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x : [batch_size, num_heads ,seq_len, head_size]
        self.inv_freq = self.inv_freq.to(x.device)
        # copying the inv freq tensor for batch in sequence
        # inv_freq_expanded : [batch_size, head_dim/2,1]
        inv_freq_expanded = self.inv_freq[None, :, None].expand(
            position_ids.shape[0], -1, -1
        )  # [batch_size, head_dim/2,1]
        # position_ids_expanded : [batch_size,1 ,seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # multiply position ids with inv freq to get the sinusoidal frequencies
            # freqs : [batch_size, head_dim/2, 1] @ [batch_size,1, seq_len] -> [batch_size,seq_len, head_dim/2]
            freqs = torch.matmul(
                inv_freq_expanded.float(), position_ids_expanded.float()
            ).transpose(1, 2)
            # embed : [batch_size, seq_len, head_dim]
            embed = torch.cat((freqs, freqs), dim=-1)
            # compute cos and sin
            cos = embed.cos().to(x.dtype)
            sin = embed.sin().to(x.dtype)
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # adding dimension for num heads
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    # building [-x2,x1,-x4,x3,...] tensor for sin part of positonal encoding.
    x1 = x[..., : x.shape[-1] // 2]  # takes first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :]  # takes second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size  # size of embedding vector of each token
        self.attention_dropout = config.attention_dropout  # we dont use
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self.rotary_emb = GemmaRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()
        # [batch_size, seq_len, hidden_size]
        # [batch_size, seq_len, num_heads*head_dim]
        query_states = self.q_proj(hidden_states)
        # [batch_size, seq_len, num_key_value_heads*head_dim]
        key_states = self.k_proj(hidden_states)
        # [batch_size, seq_len, num_key_value_heads*head_dim]
        value_states = self.v_proj(hidden_states)
        # reshape for multi head attention
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # [batch_size, num_key_value_heads, seq_len, head_dim]
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # [batch_size, num_key_value_heads, seq_len, head_dim]

        # apply rotary embeddings
        # [batch_size,seq_len, head_dim], [batch_size,seq_len, head_dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [batch_size, num_heads, seq_len, head_dim], [batch_size, num_key_value_heads, seq_len, head_dim]
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if kv_cache is not None:
            # update kv cache
            key_states, value_states = kv_cache.update(
                key_states, value_states, self.layer_idx
            )

        # we need to repeat the key and value states to match the number of heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attention_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        assert attention_mask is not None
        attention_weights = attention_weights + attention_mask

        # applying softmax
        # shape : [batch_size, num_heads, query_len, key_len]
        attention_weights = F.softmax(
            attention_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # dropout but we dont use it
        attention_weights = F.dropout(
            attention_weights, p=self.attention_dropout, training=self.training
        )
        # shape : [batch_size, num_heads, query_len, head_dim]

        # Now we need to mutliply the attention weights with value states
        attention_output = torch.matmul(attention_weights, value_states)
        # [batch_size, num_heads, seq_len_q, seq_len_kv] x [batch_size, num_heads_kv, seq_len_kv, head_dim] -> [batch_size, num_heads, seq_len_q, head_dim]

        if attention_output.size() != (
            batch_size,
            self.num_heads,
            seq_len,
            self.head_dim,
        ):
            raise ValueError("Attention output shape is incorrect")

        # make sure sequence length is the 2nd dimension
        attention_output = (
            attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        # [batch_size, seq_len_q, num_heads*head_dim]

        # now multiply with output projection
        attention_output = self.out_proj(
            attention_output
        )  # [batch_size, seq_len_q, hidden_size]

        return attention_output, attention_weights


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        (
            hidden_states,
            _,
        ) = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # [batch_size, seq_len, hidden_size]
        hidden_states = input_embeds
        # [batch_size, seq_len, hidden_size]
        normaliser = torch.tensor(
            self.config.hidden_size**0.5,
            dtype=hidden_states.dtype,
        )
        hidden_states = hidden_states * normaliser
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):  # (gemma model + linear head for lm)
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # input embeds shape : [batch_size, seq_len, hidden_size]
        # output shape : [batch_size, seq_len, hidden_size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # convert back to float32 for numerical stability
        return_data = {"logits": logits}
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        return return_data


class PaliGemmaConditionalGeneration(
    nn.Module
):  # its called conditional generation because we are conditioning generation of text on the images provided as input
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.projector = PaliGemmaMultimodalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def tie_weights(self):
        """
        Docstring for tie_weight
        technique for using parameters of one layer in another layer to reduce the number of parameters
        In our case,
        this function ties the weights of input embeddings and output embeddings of the language model
        """
        return self.language_model.tie_weights()

    def merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        _, _, embed_dim = image_features.shape
        batch_size, seq_len = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device
        # shape : [batch_size,seq_len, hidden_size]
        scaled_img_features = image_features / (self.config.hidden_size**0.5)

        # combine the embeddings of image tokens and text tokens and mask out all the padding tokens
        final_embedding = torch.zeros(
            batch_size,
            seq_len,
            embed_dim,
            dtype=input_embeds.dtype,
            device=input_embeds.device,
        )
        # Shape : [batch_size, seq_len]. True for text tokens. Something which is not image token or pad token
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )
        # input looks like : [357,357,357,357,56,432,534,54,29,2(\n)]
        # the text mask will be like : [0,0,0,0,1,1,1,1,1,1]
        # Shape : [batch_size, seq_len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # image mask will be like : [1,1,1,1,0,0,0,0,0,0]
        # Shape : [batch_size, seq_len]. True for pad tokens
        pad_mask = input_ids == self.pad_token_id
        # pad mask will be like : [0,0,0,0,0,0,0,0,0,0] since there is no padding in our input

        # now we need to expand the mask of embeddings to match the embedding dimension
        expanded_text_mask = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        expanded_image_mask = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        expanded_pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # add the text embeddings where text mask is true
        final_embedding = torch.where(expanded_text_mask, input_embeds, final_embedding)
        # whenever text mask is true take from input embeds else take from final embedding(which is zero at this point)
        # now we need to add the image embeddings where image mask is true
        final_embedding = final_embedding.masked_scatter(
            expanded_image_mask, scaled_img_features
        )
        # we didnt use torch.where because the length of scaled image features is not same as seq len of final embedding
        # so we used masked_scatter which will take values from scaled image features and put it in final embedding wherever image mask is true
        # finally we need to 0 out the pad token embeddings
        final_embedding = torch.where(
            expanded_pad_mask, torch.zeros_like(final_embedding), final_embedding
        )

        # Creation of new attention mask
        dtype, device = input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]
        if kv_cache is None or kv_cache.num_items() == 0:
            # no masking needed as we are in prefill stage
            # This only works when there is no padding in the input
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
            # we are not masking here as this is for the prefix stage where all tokens are available and we dont need to generate anything so no need for causal masking
        else:
            # since we are generating tokens , the query length is 1
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # we dont need to mask anything since each query can attend to all previous tokens
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        causal_mask = causal_mask.unsqueeze(1)  # shape : [batch_size,1, q_len, kv_len]

        if kv_cache is not None and kv_cache.num_items() > 0:
            # the position of query is just the last position
            position_ids = attention_mask.cumsum(dim=-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # create posn ids based on size of attention mask, for masked token using number 1 as postion
            position_ids = (
                (attention_mask.cumsum(dim=-1))
                .masked_fill_((attention_mask == 0), 1)
                .to(device)
            )

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """
        Docstring for forward

        here inputs ids are the token ids of the text prompt with image tokens prepended
        pixel_values are the preprocessed images
        attention_mask is the attention mask for the input ids
        kv_cache is the key value cache for the transformer model to speed up inference

        """
        assert torch.all(attention_mask == 1), "input cant be padded"
        # extract input embeddings
        # shape -> (batch,seq_len,hidden_size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        # combine text and images
        # shape- [batch,channel,height,width]->[batch,num_patches,embed_dim]
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype))
        # [batch,num_patches,embed_dim]->[batch,num_patches,hidden_size]
        image_features = self.projector(selected_image_feature)

        # merge the embeddings
        input_embeds, attention_mask, position_ids = (
            self.merge_input_ids_with_image_features(
                image_features, input_embeds, input_ids, attention_mask, kv_cache
            )
        )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )
        return outputs
