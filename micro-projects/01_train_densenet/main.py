import os
import wandb
import collections
from absl import app
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import tree_util

import optax
import jaxopt
import flax.linen as nn

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models import densenet_
from config import get_config


def setup_data(config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1.0),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1.0),
    ])
    
    train_dataset = datasets.CIFAR10(config.data_dir, download=config.download, train=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(config.data_dir, download=config.download, train=False, transform=transform_val)
    num_classes = 10
    input_shape = (32, 32, 3)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        prefetch_factor=config.loader_prefetch_factor
    )    
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        prefetch_factor=config.loader_prefetch_factor
    )
    
    return num_classes, input_shape, train_loader, val_loader


def main(_):
    config = get_config()
    wandb.init(project="learning-jax-1")
    wandb.config.update(vars(config))
    num_classes, input_shape, train_loader, val_loader = setup_data(config)
    
    norm_kwargs = lambda train: {"use_running_average": not train}
    model = densenet_(num_classes=num_classes, norm=nn.BatchNorm)
    rng_init, _ = jax.random.split(jax.random.PRNGKey(0))
    init_vars = model.init(rng_init, jnp.zeros((1,) + input_shape), norm_kwargs=norm_kwargs(train=True))
    params, batch_stats = init_vars["params"], init_vars["batch_stats"]

    print("num params: ")
    total_steps = config.epochs * len(train_loader)
    schedule = optax.cosine_decay_schedule(config.init_learning_rate, total_steps)
    optimizer = optax.sgd(schedule, momentum=.9)
    optimizer_state = optimizer.init(params)
    
    log_loss = jax.vmap(jaxopt.loss.multiclass_logistic_loss)
    
    
    def objective_fn(params, mutable_vars, images, labels):        
        model_vars = {"params": params, **mutable_vars}
        logits, mutated_vars = model.apply(
            model_vars,
            images,
            norm_kwargs=norm_kwargs(train=True),
            mutable=list(mutable_vars.keys())
        )
        loss = jnp.mean(log_loss(labels, logits))
        params_ = list(tree_util.tree_leaves(params))
        reg_term = .5 * sum(jnp.sum(jnp.square(x)) for x in params_)
        objective = loss + config.weight_decay * reg_term
        return objective, (logits, mutated_vars)

    
    @jax.jit
    def train_step(optimizer_state, params, mutable_vars, images, labels):
        objective_grad = jax.value_and_grad(objective_fn, has_aux=True)
        (objective, aux), grads = objective_grad(params, mutable_vars, images, labels)
        logits, mutated_vars = aux
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        return optimizer_state, params, mutated_vars, objective, logits
        
    @jax.jit
    def apply_model(params, batch_stats, images):
        return model.apply(
            {"params": params, "batch_stats": batch_stats}, 
            images, 
            norm_kwargs=norm_kwargs(train=False)
        )
    
    for epoch in range(1, config.epochs+1):
        metrics = collections.defaultdict(list)
        pb = tqdm(train_loader, f"epoch {epoch}/{config.epochs}:")
        for images, labels in pb:
            images, labels = jnp.asarray(images.numpy()), jnp.asarray(labels.numpy())
            images = jnp.moveaxis(images, -3, -1)
            
            optimizer_state, params, mutated_vars, objective, logits = train_step(
                optimizer_state, params, {"batch_stats": batch_stats}, images, labels
            )
            batch_stats = mutated_vars["batch_stats"]
            loss = log_loss(labels, logits)            
            preds = jnp.argmax(logits, axis=-1)
            acc = (preds == labels)
            metrics["train/acc"].append(acc)
            metrics["train/loss"].append(loss)


        # shapes = []
        # for k, v in metrics.items():
        #     v_shapes = set([v_i.shape for v_i in v])
        #     shapes.append((k, v_shapes))

        # print(shapes)
        # raise Exception()
        metrics.update({
            k: np.concatenate(v).mean()
            for k, v in metrics.items()
        })
        
 
        pb = tqdm(val_loader, "val")
        for images, labels in pb:
            images, labels = jnp.asarray(images.numpy()), jnp.asarray(labels.numpy())
            images = jnp.moveaxis(images, -3, -1)
            
            logits = apply_model(params, batch_stats, images)
            loss = log_loss(labels, logits)
            loss = log_loss(labels, logits)
            
            pred = jnp.argmax(logits, axis=-1)
            acc = (pred == labels)
            
            metrics["val/acc"].append(acc)
            metrics["val/loss"].append(loss)

        metrics.update({
            k: np.concatenate(v).mean()
            for k, v in metrics.items() if k.startswith("val")            
        })

        wandb.log(metrics)


if __name__ == "__main__":
    app.run(main)