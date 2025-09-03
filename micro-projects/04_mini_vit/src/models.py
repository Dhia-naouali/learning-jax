import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass


@dataclass
class ViTConfig:
    image_size: int = 256
    patch_size: int = 4
    num_classes: int = 100
    embed_dim: int = 384
    mlp_dim: int = 384
    num_layers: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.12
    drop_path_rate: float = 0.1
    

class PatchEmbed(nn.Module):
    patch_size: int
    embed_dim: int
    
    @nn.compact
    def __call__(self, x):
        p = (self.patch_size, self.patch_size)
        b, h, w, c = x.shape
        assert not h % self.patch_size and not w % self.patch_size, \
            f"invalid input shape {x.shape}, patch_size: {self.patch_size}"
        
        x = nn.Conv(
            self.embed_dim, 
            kernel_size=p,
            strides=p,
        )(x)
        return x.reshape(b, -1, self.embed_dim)


class RoPEAttention(nn.Module):
    num_heads: int
    embed_dim: int
    dropout_rate: float = .1
    
    @nn.compact
    def __call__(self, x, deterministic):
        head_dim = self.embed_dim // self.num_heads
        b, n, d = x.shape
        qkv = nn.Dense(self.embed_dim * 3)(x)
        # b, n, emb_dim
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # b, h, n, hd
        q, k, v = [
            t.reshape(b, n, self.num_heads, head_dim).transpose(0, 2, 1, 3)
            for t in [q, k, v]
        ]
        
        q, k = self.apply_rope(q, k)
        
        # b, h, emb_dim, emb_dim
        attn_scores = nn.softmax(
            jnp.einsum("bhqd, bhkd -> bhqk", q, k) / jnp.sqrt(head_dim),
            axis=-1 # along the query axis
        )
        
        attn_scores = nn.Dropout(self.dropout_rate)(
            attn_scores, 
            deterministic=deterministic,
            rng=None if deterministic else self.make_rng("dropout")
        )
        attn = jnp.einsum("bhqk, bhvd -> bhqd", attn_scores, v) # leading with the query dim

        # back to b, n, emb_dim
        attn = attn.transpose(0, 2, 1, 3).reshape(b, n, self.embed_dim)
        return nn.Dense(self.embed_dim)(attn)

    def apply_rope(self, q, k):
        b, h, n, hd = q.shape
        assert hd % 2 == 0, "head dim should be even when using RoPE"
        hd2 = hd // 2
        theta = 1e4 ** (-2 * jnp.arange(hd2) / hd)
        pos = jnp.arange(n)
        freqs = jnp.einsum("i, j -> ij", pos, theta)

        def rope_rotate(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            return jnp.concatenate([ # rotation matrix
                cos * x1 - sin * x2,
                sin * x1 + cos * x2
            ], axis=-1)

        # 1, 1, n, hd/2
        cos = jnp.cos(freqs)[None, None, :, :]
        sin = jnp.sin(freqs)[None, None, :, :]

        return rope_rotate(q), rope_rotate(k)



class MLP(nn.Module):
    hidden_dim: int
    out_dim: int
    dropout_rate: float = .1
    
    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic, rng=None if deterministic else self.make_rng("dropout"))
        x = nn.Dense(self.out_dim)(x)
        return x



class DropPath(nn.Module):
    drop_rate: float
    
    def _drop_path(self, x, deterministic):
        rng = jax.random.key(12) if deterministic else self.make_rng("drop_path")
        keep_prob = 1 - self.drop_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=shape).astype(x.dtype)
        return x * mask / keep_prob
        

    @nn.compact
    def __call__(self, x, deterministic):
        return jax.lax.cond(
            deterministic | (self.drop_rate == 0),
            lambda x: x,
            lambda x: self._drop_path(x, deterministic),
            x
        )


class ViTBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = .1
    drop_path_rate: float = .1
    
    @nn.compact
    def __call__(self, x, deterministic):
        
        x_res = x
        x = nn.LayerNorm()(x)
        x = RoPEAttention(
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            dropout_rate=self.dropout_rate
        )(x, deterministic=deterministic)
        x = DropPath(self.drop_path_rate)(x, deterministic=deterministic)
        x_res = x + x_res
        
        x = nn.LayerNorm()(x)
        x = MLP(
            hidden_dim=self.mlp_dim,
            out_dim=self.embed_dim,
            dropout_rate=self.dropout_rate
        )(x, deterministic=deterministic)
        x = DropPath(self.drop_path_rate)(x, deterministic=deterministic)
        
        return x + x_res



class ViT(nn.Module):
    config: ViTConfig
    
    def setup(self):
        self.patch_embed = PatchEmbed(
            self.config.patch_size, self.config.embed_dim
        )
        self.cls_token = self.param(
            "cls_token", 
            nn.initializers.normal(stddev=.02), 
            (1, 1, self.config.embed_dim)
        )
        
        dropout_rates = jnp.linspace(0, self.config.drop_path_rate, self.config.num_layers, dtype=jnp.float32)
        self.blocks = [
            ViTBlock(
                embed_dim=self.config.embed_dim,
                mlp_dim=self.config.mlp_dim,
                num_heads=self.config.num_heads,
                dropout_rate=self.config.dropout_rate,
                drop_path_rate=dpr
            ) for dpr in dropout_rates
        ]
        
        self.norm = nn.LayerNorm()
        self.head = nn.Dense(self.config.num_classes)
        
        
    def __call__(self, x, deterministic):        
        x = self.patch_embed(x)
        cls_token = jnp.tile(self.cls_token, (x.shape[0], 1, 1))
        x = jnp.concatenate([cls_token, x], axis=1)
        
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)