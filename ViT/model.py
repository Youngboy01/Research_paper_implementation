import torch
import torch.nn as nn
# THE WHOLE IMPLEMENTATION IS BASED ON THE BASE VIT ARCHITECTURE
# https://arxiv.org/abs/2010.11929
class PatchEmbeddings(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channel=3,
        base_embedding_dim=768,
        bias=True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.base_embedding_dim = base_embedding_dim

        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

        self.proj = nn.Conv2d(
            in_channel,
            base_embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )  # coz no overlap

    def forward(self, x):
        print(x.shape)
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        print(x.shape)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, attn_p=0.0, proj_p=0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # The data is laid out as [Q1, K1, V1, Q2, K2, V2, ...] per token, not as [Q_all, K_all, V_all] blocks. So if you try to reshape directly into [3, batch_size, num_heads, seq_len, head_dim], you're assuming the data is already grouped by Q/K/V — which it’s not.
        q, k, v = qkv[0], qkv[1], qkv[2]  # each has shape (B,num_heads,N,head_dim)
        score = (
            (q @ k.transpose(-2, -1)) * self.scale
        )  # (B,num_heads,N,head_dim) @ (B,num_heads,head_dim,N) -> (B,num_heads,N,N)
        attn = score.softmax(dim=-1)
        attn = self.attn_drop(attn)  # (B,num_heads,N,N)
        out = (
            attn @ v
        )  # (B,num_heads,N,N) @ (B,num_heads,N,head_dim) -> (B,num_heads,N,head_dim)
        out = out.transpose(1, 2)  # (B,N,num_heads,head_dim)
        out = out.flatten(2)  # (B,N,embed_dim)
        out = self.proj(out)  # (B,N,embed_dim)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, embed_dim=768, mlp_dim=3072, dropout_p=0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.mlp(x)


# class LayerNorm(nn.Module):
#     def __init__(self, eps=1e-6, normalised_shape=768):
#         super().__init__()
#         self.eps = eps
#         self.gamma = nn.Parameter(torch.ones(normalised_shape))
#         self.beta = nn.Parameter(torch.zeros(normalised_shape))

#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.gamma * (x - mean) / (std + self.eps) + self.beta
# nn.LayerNorm does the same thing as above custom LayerNorm class but in a more optimized way


class TranformerBlock(nn.Module):
    def __init__(self, num_heads, embed_dim, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, attn_p=dropout, proj_p=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, mlp_dim=mlp_dim, dropout_p=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # RESIDUAL CONNECTION
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                TranformerBlock(num_heads, embed_dim, mlp_dim, dropout)
                for _ in range(12)
            ]
        )

    # nn.Sequential constructor expects its arguments to be individual nn.Modules, not a single list containing all the modules.
    def forward(self, x):
        return self.blocks(x)


# SINCE WE ARE TRYING OUR MODEL ON IMAGE NET SO MAKE NUM_CLASSES=1000
class ViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channel=3,
        embed_dim=768,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
        num_classes=1000,
    ) -> None:
        super().__init__()
        self.patch_embeds = PatchEmbeddings(
            image_size, patch_size, in_channel, embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.position_embeds = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embeds.num_patches, embed_dim)
        )
        self.encoder = Encoder(embed_dim, num_heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embeds(x)  # (B,num_patches,embed_dim)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B,1+num_patches,embed_dim)
        x = x + self.position_embeds  # (B,1+num_patches,embed_dim)
        x = self.dropout(x)
        x = self.encoder(x)  # (B,1+num_patches,embed_dim)
        x = self.norm(x)
        cls_token_final = x[:, 0]  # (B,embed_dim)
        x = self.head(cls_token_final)  # (B,num_classes)
        return x
