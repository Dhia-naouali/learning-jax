import jax
import functools


def make_sharding(axis_name="data"):# oh how different
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices), (axis_name,))
    partition = jax.sharding.PartitionSpec(axis_name)
    sharding = jax.shardin.NamedSharding(mesh, partition)
    return sharding


class RecordWriter:
    prev_metric = None
    
    def __call__(self, cur_metrics):
        self.prev_metrics, log_metrics = cur_metrics, self.prev_metrics
        if log_metrics is None:
            return
        print(", ".join([f"{k}: {float(v):.4f}" for k, v in log_metrics.items()]))



def shard_batch(batch, num_devices):
    images = batch["images"]
    labels = batch["labels"]
    assert images.shape[0] % num_devices == 0
    local_b = images.shape[0] // num_devices

    images = images.reshape(
        (num_devices, local_b) + images.shape[1:]
    )
    labels = labels.reshape(
        (num_devices, local_b) + labels.shape[1:]
    )

    return images, labels



# def shard_batch(images, labels):
#     world_size = jax.device_count() # local_device_count if I had multi hosts (delulu~)
#     assert images.shape[0] % world_size == 0
#     device_batch = images.shape[0] // world_size
#     images = images.reshape((world_size, device_batch) + images.shape[1:])
#     labels = labels.reshape((world_size, device_batch) + labels.shape[1:])
#     return images, labels


# def unreplicate_state(replicated):
#     return jax.device_get(jax.tree_map(lambda x: x[0], replicated))