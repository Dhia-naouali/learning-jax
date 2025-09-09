import jax

def shard_batch(images, labels):
    world_size = jax.device_count() # local_device_count if I had multi hosts (delulu~)
    assert images.shape[0] % world_size == 0
    device_batch = images.shape[0] // world_size
    images = images.reshape((world_size, device_batch) + images.shape[1:])
    labels = labels.reshape((world_size, device_batch) + labels.shape[1:])
    return images, labels

def unreplicate_state(replicated):
    return jax.device_get(jax.tree_map(lambda x: x[0], replicated))