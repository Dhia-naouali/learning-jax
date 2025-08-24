import math
from functools import partial
from typing import Callable, Sequence

from flax import linen as nn
from jax import numpy as jnp

ModuleDef = Callable[..., nn.Module]


class Bottleneck(nn.Module):
    growth_rate: int
    norm: ModuleDef

    def setup(self):
        self.bn1 = self.norm()
        self.conv1 = nn.Conv(4*self.growth_rate, (1, 1), use_bias=False)
        self.bn2 = self.norm()
        self.conv2 = nn.Conv(self.growth_rate, (3, 3), padding=(1, 1), use_bias=False)

    def __call__(self, x, norm_kwargs=None):
        norm_kwargs = norm_kwargs or {}
        out = self.conv1(nn.relu(self.bn1(x, **norm_kwargs)))
        out = self.conv2(nn.relu(self.bn2(out, **norm_kwargs)))
        return jnp.concatenate([out, x], axis=-1)



class Transition(nn.Module):
    out_planes: int
    norm: ModuleDef
    
    def setup(self):
        self.bn = self.norm()
        self.conv = nn.Conv(self.out_planes, (1, 1), use_bias=False)
    
    def __call__(self, x, norm_kwargs=None):
        norm_kwargs = norm_kwargs or {}
        x = self.conv(nn.relu(self.bn(x, **norm_kwargs)))
        return nn.avg_pool(x, (2, 2), strides=(2, 2), padding="VALID")



class BlockSequence(nn.Module):
    block: ModuleDef
    growth_rate: int
    nblocks: int
    norm: ModuleDef = nn.BatchNorm
    
    @nn.compact
    def __call__(self, x, norm_kwargs=None):
        for i in range(self.nblocks):
            block = self.block(self.growth_rate, norm=self.norm, name=f"block_{i}")
            x = block(x, norm_kwargs=norm_kwargs)
        return x



class DenseNet(nn.Module):
    block: ModuleDef
    nblocks: Sequence[int]
    growth_rate: int
    num_classes: int
    norm: ModuleDef
    reduction: float = .5

    def setup(self):
        make_dense_layers = partial(
            BlockSequence,
            self.block,
            self.growth_rate,
            norm=self.norm
        )
    
        num_planes = 2 * self.growth_rate
        self.conv1 = nn.Conv(
            num_planes, (3, 3), padding=(1, 1), use_bias=False
        )
        
        self.dense1 = make_dense_layers(self.nblocks[0])
        num_planes += self.nblocks[0] * self.growth_rate
        out_planes = int(math.floor(num_planes*self.reduction))
        self.trans1 = Transition(out_planes, norm=self.norm)
        num_planes = out_planes

        self.dense2 = make_dense_layers(self.nblocks[1])
        num_planes += self.nblocks[1]*self.growth_rate
        out_planes = int(math.floor(num_planes*self.reduction))
        self.trans2 = Transition(out_planes, norm=self.norm)
        num_planes = out_planes

        self.dense3 = make_dense_layers(self.nblocks[2])
        num_planes += self.nblocks[2]*self.growth_rate
        out_planes = int(math.floor(num_planes*self.reduction))
        self.trans3 = Transition(out_planes, norm=self.norm)
        num_planes = out_planes        
        
        self.dense4 = make_dense_layers(self.nblocks[3])
        num_planes += self.nblocks[3]*self.growth_rate
        
        self.bn = self.norm()
        self.fc = nn.Dense(self.num_classes)
        
    def __call__(self, x, norm_kwargs=None):
        nk = norm_kwargs or {}
        x = self.conv1(x)
        x = self.trans1(self.dense1(x, norm_kwargs=nk), norm_kwargs=nk)
        x = self.trans2(self.dense2(x, norm_kwargs=nk), norm_kwargs=nk)
        x = self.trans3(self.dense3(x, norm_kwargs=nk), norm_kwargs=nk)
        x = self.dense4(x, norm_kwargs=nk)
        x = nn.relu(self.bn(x, **nk))
        x = nn.avg_pool(x, (4, 4), strides=(4, 4), padding="VALID")
        x = jnp.reshape(x, (*x.shape[:-3], -1)) # bs?, -1
        return self.fc(x)

densenet_ = partial(DenseNet, Bottleneck, (6, 12, 24, 16), growth_rate=12)