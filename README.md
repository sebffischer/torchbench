TODOs:

* better understand why R is slower than python:
    * I don't think the GC is so much of a problem anymore
    * We need to benchmark individual steps like the forward pass, backward pass and optimizer

* benchmark for dataset
* benchmark for dataloader
* benchmark for optimizers 
* create a website for this
* run everything in a container


Some things to pay attention to:

* set number of threads for CPU.
* synchronize the cuda streams before starting the measurement and before ending it.
* set CUDA_VISIBLE_DEVICES if you have more than one GPU so that always the same one will be used.

