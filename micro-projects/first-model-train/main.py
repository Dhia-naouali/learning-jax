import os
import tqdm
import wandb
import collections
from ml_collections import config_flags, config_dict

from absl import flags, app
import jax
import jax.numpy as jnp
from jax import tree_util
import flax.linen as nn

import jaxopt
import optax

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models import densenet_

flags.DEFINE_string("dataset_root", "data", "path to data directory")
flags.DEFINE_bool("download", True, "download dataset")
flags.DEFINE_integer("loader_prefetch_factor", 2, "prefetch factor for loader")

flags.DEFINE_integer("train_batch_size", 32, "")
flags.DEFINE_integer("eval_batch_size", 128, "")
flags.DEFINE_integer("epochs", 32, "")
flags.DEFINE_float("init_learning_rate", 3e-4, "")
flags.DEFINE_float("weight_decay", 1e-2, "")
config_flags.DEFINE_config_file("config")


# train_batch_size, val_batch_size, epochs, init_learning_rate, weight_decay


FLAGS = flags.FLAGS

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
    
    train_dataset = datasets.CIFAR10(FLAGS.dataset_root, download=FLAGS.download, train=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(FLAGS.dataset_root, download=FLAGS.download, train=False, transform=transform_val)
    num_classes = 10
    input_shape = (32, 32, 3)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        prefetch_factor=FLAGS.loader_prefetch_fator
    )
    
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        prefetch_factor=FLAGS.loader_prefetch_fator
    )
    
    return num_classes, input_shape, train_loader, val_loader



def make_model(num_classes, norm=nn.BatchNorm):
    return densenet_(num_classes, norm)

def main(_):
    config = config_dict.ConfigDict(FLAGS.config)
    wandb.init(project="learning-jax-1")
    wandb.config.update(config.to_dict())
    num_classes, input_shape, train_loader, val_loader = setup_data(config)
    
    model = make_model(config, num_classes, input_shape)
    rng_init, _ = jax.random.split(jax.random.PRNGKey(0))
    params = model.init(rng_init, jnp.zeros((1,) + input_shape))
    
    total_steps = config.epochs * len(train_loader)
    schedule = optax.cosine_decay_schedule(config.init_learning_rate, total_steps)
    optimizer = optax.sgd(schedule, momentum=.9)
    optimizer_state = optimizer.init(params)
    
    log_loss = jax.vmap(jaxopt.loss.multiclass_logistic_loss)
    
    def objective_fn(labels, logits):
        loss = jnp.mean(log_loss(labels, logits))    
        wd_params = tree_util.tree_leaves(params)
        reg_term = .5 * sum(jnp.sum(jnp.square(x) for x in wd_params))
        objective = loss + config.weight_decay * reg_term
        return objective
    
    @jax.jit
    def train_step(optimizer_state, params, data):
        inputs, labels = data
        logits = model.apply(params, inputs)
        loss = objective_fn(labels, logits)
        grads = jax.grad(objective_fn)(labels, logits)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        return optimizer_state, params, logits, loss
    

    for epoch in range(1, config.epochs+1):
        
        metrics = collections.defaultdict(list)
        pb = tqdm(train_loader, f"epoch {epoch}/{config.epochs}:")
        for inputs, labels in pb:
            inputs, labels = jnp.asarray(inputs.numpy()), jnp.asarray(labels.numpy())
            inputs = jnp.moveaxis(inputs, -3, -1)
            optimizer_state, params, logits, loss = train_step(
                optimizer_state, params, (inputs, labels)
            )
            
            preds = jnp.argmax(logits)
            acc = (preds == logits)
            metrics["train/acc"].append(acc)
            metrics["train/loss"].append(loss)

        metrics.update({
            k: np.array(v).mean() 
            for k, v in metrics.items() if k.startswith("train")
        })
        
 
        pb = tqdm(val_loader, "val")
        for inputs, labels in pb:
            inputs, labels = jnp.asarray(inputs.numpy()), jnp.asarray(labels.numpy())
            inputs = jnp.moveaxis(inputs, -3, -1)
            logits = model.apply(params, inputs)
            pred = jnp.argmax(logits, axis=-1)
            loss = objective_fn(labels, logits)
            acc = (pred == labels)
            metrics["val/acc"].append(acc)
            metrics["val/loss"].append(loss)

        metrics.update({
            k: np.array(v).mean()
            for k, v in metrics.items() if k.startswith("val")
        })


if __name__ == "__main__":
    app.run(main)