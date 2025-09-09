# 🗣️ Hold my state

<p align="center">
  <img src="assets/woody.gif" width="420"><br>
  <em> just grab some other toy now com'on</em>
</p>


| micro-project | Main Learnings | viz |
|-------------|----------------|------------------------|
| [01_train_densenet](micro-projects/01_train_densenet) |  - JAX basics: <br> &nbsp;&nbsp;- stateless mess <br> &nbsp;&nbsp;- grad <br> &nbsp;&nbsp;- jit <br> &nbsp;&nbsp;- model def <br> &nbsp;&nbsp;- ... <br> - tossing batch norm running stats around  | <div align="center"><img height="200" alt="image" src="https://github.com/user-attachments/assets/b576623e-ae44-4b03-b1f4-849f937d9b87" /><div> |
| [02_nn_mnist](micro-projects/02_nn_mnist) | getting used to JAX syntax | <div align="center"><img height="200" alt="image" src="https://github.com/user-attachments/assets/c65be976-8f15-437d-86b8-74250464acb2" /> <div> |
| [03_logistic_regression](micro-projects/03_logistic_regression) | classic ML (filling a quick break) | <div align="center"><img height="200" height="1425" alt="3 1" src="https://github.com/user-attachments/assets/6b505af6-0fb0-4fc7-b235-c242168a4311" /> <div> |
| [04_mini_vit](micro-projects/04_mini_vit) <br> **(to fix)** | vision transformer in jax <br> - getting used to @nn.compact <br> - deterministic mess <br> - RoPE embedding | <div align="center"><img height="200" height="1425" alt="3 1" src="https://github.com/user-attachments/assets/c6508491-8b0c-4cfa-b1eb-3b8b263a26c0" /> <div> |
| [05_distributed_training](micro-projects/05_distributed_training) <br> **(to optimize)** | distributed training of a simple CNN on CIFAR10 | ... |




| the thingy I tried | Main Learnings |
|-------------|----------------|
| [spmd_linear](trying-things-out/spmd_linear.py) | FSDP linear layer |
| [step_bubble_benchmark](trying-things-out/step_bubble_benchmark.py) | asynchronous dispatch flow (load data batch & compute in ~parallel)|