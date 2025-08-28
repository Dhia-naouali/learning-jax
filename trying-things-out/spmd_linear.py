import jax
from jax.sharding import PartitionSpec as P
import numpy as np

jax.config.update("jax_num_cpu_devices", 4)
key = jax.random.key(12)
key, x_key, w_key, b_key = jax.random.split(key, 4)

@jax.jit
def linear(x, w, b):
    print(f"{jax.typeof(x) = }")
    print(f"{jax.typeof(w) = }")
    print(f"{jax.typeof(b) = }")
    return x @ w + b

x = jax.random.uniform(x_key, (256, 32))
w = jax.random.uniform(w_key, (32, 16))
b = jax.random.uniform(b_key, (16,))
x_np = np.array(x)
w_np = np.array(w)
b_np = np.array(b)

mesh = jax.make_mesh((4,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,))
sharding = jax.NamedSharding(mesh, P("x"))
replicated_sharding = jax.NamedSharding(mesh, P())


sharded_x = jax.device_put(x, sharding)
sharded_w = jax.device_put(w, replicated_sharding)
sharded_b = jax.device_put(b, replicated_sharding)


with jax.set_mesh(mesh):
    y = linear(sharded_x, w, sharded_b)
    
    
print(f"{y.shape = }")
print(f"equal to single device results: {(np.array(y) == (x_np @ w_np + b)).all()}")