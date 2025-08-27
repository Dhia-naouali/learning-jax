import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import NamedTuple

class Params(NamedTuple):
    weight: jnp.ndarray
    bias: jnp.ndarray
    
def init(rng):
    w_key, b_key = jax.random.split(rng)
    w = jax.random.normal(w_key, ())
    b = jax.random.normal(b_key, ())
    return Params(w, b)

def loss(params, x, y):
    y_hat = x * params.weight + params.bias
    return jnp.mean((y - y_hat) ** 2)

learning_rate = 5e-2

@jax.jit
def update(params, x, y):
    grads = jax.grad(loss)(params, x, y)
    return jax.tree.map(
        lambda g, p: p - learning_rate*g,
        grads,
        params
    )


w_ = 1.6
b_ = .8
n = 128

key = jax.random.key(12)
key, params_key, x_key, y_key = jax.random.split(key, 4)

xs = jax.random.normal(x_key, (n, 1))
noise = jax.random.normal(y_key, (n, 1))*0.4
ys = xs * w_ + b_ + noise

params = init(params_key)

for _ in range(32):
    params = update(params, xs, ys)

y_pred = xs * params.weight + params.bias
plt.scatter(xs, ys)
plt.plot(xs, y_pred, c="red")
plt.show()