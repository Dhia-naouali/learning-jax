import time
import jax
import jax.numpy as jnp


def batch_generator(num_batches=500, seed=12, shape=(32, 2**7, 2**7)):
    key = jax.random.PRNGKey(seed)
    for _ in range(num_batches):
        key, sub = jax.random.split(key)
        yield jax.random.normal(sub, shape)


@jax.jit
def compute(data):
    res = jnp.pow(data, 2)
    norm = jnp.linalg.norm(data, ord=2, axis=-1, keepdims=True)
    data = data * jax.lax.rsqrt(norm)
    for _ in range(20):
        data = data @ res
    return jnp.log(jnp.abs(data) + 1e-6)


def bench_with_breaker(num_batches=500, breaker=lambda x: None):
    gen = batch_generator(num_batches)
    start = time.perf_counter()
    for batch in gen:
        res = compute(batch)
        breaker(res)

    res.block_until_ready() # all compts done
    end = time.perf_counter()
    return (end - start) / num_batches


none_breaker = lambda x: None
shape_breaker = lambda x: x.shape
type_breaker = lambda x: x.dtype
per_iter_block = lambda x: x.block_until_ready()
read_element = lambda x: x.at[..., 0].get()
read_slice = lambda x: x.at[0:2, 0:2, 0:2].get()

sample = next(batch_generator(1))
compute(sample).block_until_ready()

breakers = [
    "none_breaker",
    "shape_breaker",
    "type_breaker",
    "per_iter_block",
    "read_element",
    "read_slice",
]

for breaker in breakers:
    t = bench_with_breaker(breaker=locals()[breaker])
    print(f"{breaker}: {t*1e3:.4f} ms")