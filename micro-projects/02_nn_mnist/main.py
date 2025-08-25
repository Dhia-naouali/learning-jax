import os
import wandb
import numpy as np
import collections
from tqdm import tqdm

import jax
import optax
import jaxopt
import jax.numpy as jnp

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from model import NN

def setup_data():
    train_dataset = datasets.MNIST("data", download=True, train=True, transform=transforms.ToTensor())
    val_dataset = datasets.MNIST("data", download=True, train=False, transform=transforms.ToTensor())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=os.cpu_count()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=os.cpu_count()
    )
    
    return train_loader, val_loader, (784,), 10


def main():
    wandb.init(project="learning-jax-2")
    train_loader, val_loader, input_shape, num_classes = setup_data()
    model = NN(hidden_dim=64, num_classes=num_classes)
    rng_init, _ = jax.random.split(jax.random.PRNGKey(0))
    params = model.init(rng_init, jnp.zeros((1,) + input_shape))
    optimizer = optax.sgd(1e-3)
    optimizer_state = optimizer.init(params)
    log_loss = jax.vmap(jaxopt.loss.multiclass_logistic_loss)
    
    def loss_fn(params, images, labels):
        logits = model.apply(params, images)
        loss = jnp.mean(log_loss(labels, logits))
        return loss, logits
    
    @jax.jit
    def train_step(params, optimizer_state, images, labels):
        grads, logits = jax.grad(loss_fn, has_aux=True)(params, images, labels)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, logits
    
    for epoch in range(1, 13):
        metrics = collections.defaultdict(list)
        pb = tqdm(train_loader, f"epoch {epoch}/{12}")
        for images, labels in pb:
            images, labels = jnp.asarray(images.numpy()), jnp.asarray(labels.numpy())
            images = jnp.reshape(images, (-1, 784))
            params, optimizer_state, logits = train_step(params, optimizer_state, images, labels)
            loss = log_loss(labels, logits)
            preds = jnp.argmax(logits, axis=-1)
            acc = (preds == labels)
            metrics["train_acc"].append(acc)
            metrics["train_loss"].append(loss)
            
        metrics.update({
            k: np.concatenate(v).mean()
            for k, v in metrics.items()
        })
            
        pb = tqdm(val_loader, "val")
        for images, labels in pb:
            images, labels = jnp.asarray(images.numpy()), jnp.asarray(labels.numpy())
            images = jnp.reshape(images, (-1, 784))
            logits = model.apply(params, images)
            loss = log_loss(labels, logits)
            
            preds = jnp.argmax(logits, axis=-1)
            acc = (preds == labels)
            metrics["val_acc"].append(acc)
            metrics["val_loss"].append(loss)
            
        metrics.update({
            
            k: np.concatenate(v).mean()
            for k, v in metrics.items() if k.startswith("val")   
        })
        
        wandb.log(metrics)
        
if __name__ == "__main__":
    main()
