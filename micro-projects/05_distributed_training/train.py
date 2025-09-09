import os
import time
from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import flax

from data_loader import make_loader, transform_batch
from model import CNN
from utils import shard_batch, unreplicate_state

class TrainState(train_state.TrainState):
    pass

def create_train_state(rng, model, learning_rate):
    params = model.init(
        rng, 
        jnp.ones((1, 32, 32, 3))
    )["params"]
    optimizer = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


@partial(jax.pmap, axis_name='batch')
def train_step(state, images, labels):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        loss = optax.softmax_cross_entropy(
            logits, 
            jax.nn.one_hot(labels, num_classes=10)
        ).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # devices aggregation
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)

    # to optimizer
    metrics = {
        "loss": loss,
        "accuracy": jnp.mean(jnp.argmax(logits, -1) == labels)
    }
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return new_state, metrics

def main(
    device_batch=256, # TPU cores are chunky
    epochs=3,
    learning_rate=1e-3,
    ckpt_dir="./checkpoints",
    num_workers=None,
):
    ckpt_dir = os.path.abspath(ckpt_dir)
    world_size = jax.local_device_count()
    global_batch = device_batch * world_size

    loader = make_loader(global_batch, train=True, num_workers=num_workers)

    model = CNN(num_classes=10)
    rng = jax.random.PRNGKey(12)
    state = create_train_state(rng, model, learning_rate)

    state = checkpoints.restore_checkpoint(ckpt_dir, target=state)
    state = flax.jax_utils.replicate(state)

    step = 0
    for epoch in range(epochs):
        start = time.time()
        pb = tqdm(loader, desc=f"epoch {epoch}/{epochs}")
        for batch in pb:
            images, labels = transform_batch(batch)
            images_sharded, labels_sharded = shard_batch(images, labels)
            images = jax.device_put(images_sharded)
            labels = jax.device_put(labels_sharded)

            state, metrics = train_step(state, images, labels)
            if step % 40 == 0:
                logs = jax.tree_map(lambda x: float(x.mean()), metrics)
                pb.set_postfix(loss=f"{logs['loss']:.4f}", acc=f"{logs['accuracy']:.4f}")
            step += 1

    state_to_save = unreplicate_state(state)
    checkpoints.save_checkpoint(ckpt_dir, state_to_save, step=step, overwrite=True)
    print(f"{((time.time() - start) / epochs):.1f}s per epoch")


if __name__ == "__main__":
    main()