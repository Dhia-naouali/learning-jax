import time
import wandb
import argparse
import collections
import numpy as np
from tqdm import tqdm

import jax
import optax
import jaxopt
import jax.numpy as jnp
from flax.training import checkpoints

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
    
    main_key, dp_key, pd_key = jax.random.split(main_key, 3)
    params = model.init(
        {
            "params": init_key,
            "dropout": dp_key,
            "drop_path": pd_key
        },
        jnp.zeros((1, config.image_size, config.image_size, 3)),
        deterministic=False,
    )["params"]

    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=args.wd)
    optimizer_state = optimizer.init(params)
    
    def loss_fn(params, images, labels, rng):
        dp_key, pd_key = jax.random.split(rng)
        rngs = {
            "dropout": dp_key,
            "drop_path": pd_key
        }
        logits = model.apply(
            {"params": params},
            images,
            deterministic=False, 
            rngs=rngs
        )
        loss = jaxopt.loss.multiclass_logistic_loss(labels, logits).mean()
        return loss, logits
    
    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(params, optimizer_state, images, labels, rng):
        (loss, logits), grads = loss_grad_fn(params, images, labels, rng)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params=params)
        params = optax.apply_updates(params, updates)
        
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        return params, optimizer_state, loss, acc
    
    @jax.jit
    def val_step(params, images, labels):
        logits = model.apply({"params": params}, images, deterministic=True)
        loss = jaxopt.loss.multiclass_logistic_loss(labels, logits).mean()
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        return loss, acc

    wandb.init(project="learning-jax-4-ViT", config=vars(config))
    for epoch in range(1, args.epochs+1):
        epoch_start = time.time()
        metrics = collections.defaultdict(list)
        for images, labels in tqdm(train_loader):
            images = jnp.array(np.array(images)).astype(jnp.float32)
            labels = jnp.array(np.array(labels)).astype(jnp.int32)
            images = jnp.moveaxis(images, -3, -1)
            main_key, train_key = jax.random.split(main_key)
            params, optimizer_state, loss, acc = train_step(
                params, 
                optimizer_state,
                images,
                labels,
                train_key
            )
            metrics["train/loss"].append(float(loss))
            metrics["train/acc"].append(float(acc))

        metrics = {
            k: jnp.array(v).mean() for k, v in metrics.items()
        }
        metrics["train/time"] = time.time() - epoch_start
        wandb.log(metrics)


        metrics = collections.defaultdict(list)
        for images, labels in tqdm(val_loader):
            images = jnp.array(np.array(images)).astype(jnp.float32)
            labels = jnp.array(np.array(labels)).astype(jnp.int32)
            images = jnp.moveaxis(images, -3, -1)
            loss, acc = val_step(params, images, labels)
            metrics["val/loss"].append(float(loss))
            metrics["val/acc"].append(float(acc))
            
        metrics = {
            k: jnp.array(v).mean() for k, v in metrics.items()
        }
        wandb.log(metrics)  


if __name__ == "__main__":
    main()