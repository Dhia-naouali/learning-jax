import time
import wandb
import argparse
import collections
from tqdm import tqdm
from functools import partial

import jax
import optax
import jaxopt
import jax.numpy as jnp
from flax.training import checkpoints
from flax.training.train_state import TrainState

from models import ViT, ViTConfig
from data import make_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=5e-2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=100)

    args = parser.parse_args()
    
    config = ViTConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
    )
    
    model = ViT(config)
    train_loader, val_loader = make_loader(args.data_dir, args.batch_size)
    
    main_key = jax.random.PRNGKey(12)
    main_key, init_key = jax.random.split(main_key)
    input_shape = (1, config.image_size, config.image_size, 3)
    
    main_key, dp_key, pd_key = jax.random.split(main_key, 3)
    params = model.init(
        init_key, 
        jnp.zeros(input_shape), 
        deterministic=True,
        rngs = {
                "dropout": dp_key,
                "path_drop": pd_key
        }
    )
    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=args.wd)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

    # loss thingies
    log_loss = jax.vmap(jaxopt.loss.multiclass_logistic_loss)
    
    def loss_fn(params, images, labels, rngs: dict=None):
        logits = state.apply_fn(params, images, deterministic=rngs is None, rngs=rngs)
        loss = log_loss(logits, labels).mean()
        return loss, logits

    loss_grad = jax.value_and_grad(loss_fn, has_aux=True)
    
    wandb.init(project="learning-jax-4-ViT", config=vars(config))
    for epoch in range(1, args.epochs+1):
        epoch_start = time.time()
        metrics = collections.defaultdict(list)
        for images, labels in tqdm(train_loader):
            images, labels = jnp.asarray(images.numpy()), jnp.asarray(labels.numpy())
            images = jnp.moveaxis(images, -3, -1)

            main_key, dp_key, pd_key = jax.random.split(main_key, 3)
            rngs = {
                "dropout": dp_key,
                "path_drop": pd_key
            }
            state, loss, acc = train_step(state, images, labels, rngs, loss_grad)
            metrics["train/loss"].append(float(loss))
            metrics["train/acc"].append(float(acc))
        
        metrics = {
            k: jnp.array(v).mean() for k, v in metrics.items()
        }
        metrics["train/time"] = time.time() - epoch_start
        wandb.log(metrics)


        epoch_start = time.time()
        metrics = collections.defaultdict(list)
        for images, labels in tqdm(val_loader):
            images, labels = jnp.asarray(images.numpy()), jnp.asarray(labels.numpy())
            images = jnp.moveaxis(images, -3, -1)

            loss, acc = val_step(state, images, labels, log_loss)
            metrics["val/loss"].append(float(loss))
            metrics["val/acc"].append(float(acc))
        
        metrics = {
            k: jnp.array(v).mean() for k, v in metrics.items()
        }
        metrics["val/time"] = time.time() - epoch_start
        wandb.log(metrics)    


@partial(jax.jit, static_argnums=(4,))
def train_step(state, images, labels, rngs, loss_grad):
    (loss, logits), grads = loss_grad(state.params, images, labels, rngs=rngs)
    state = state.apply_gradients(grads)
    preds = jnp.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return state, loss, acc


@jax.jit
def val_step(state, images, labels):
    logits = state.apply_fn(state.params, images, deterministic=True, rngs=None)
    loss = jaxopt.loss.multiclass_logistic_loss(logits, labels).mean()
    preds = jnp.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return loss, acc



if __name__ == "__main__":
    main()