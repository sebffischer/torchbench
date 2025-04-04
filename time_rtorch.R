time_rtorch = function(epochs, batch_size, n_layers, latent, n, p, device, jit, seed, optimizer, mlr3torch = FALSE) {
  library(mlr3torch)
  library(torch)
  torch_set_num_threads(1)
  torch_manual_seed(seed)

  lr = 0.001

  make_network = function(p, latent, n_layers) {
    layers = list(nn_linear(p, latent), nn_relu())
    for (i in seq_len(n_layers - 1)) {
        layers = c(layers, list(nn_linear(latent, latent), nn_relu()))
    }
    layers = c(layers, list(nn_linear(latent, 1)))

    net = do.call(nn_sequential, args = layers)
    net
  }


  X = torch_randn(n, p, device = device)
  beta = torch_randn(p, 1, device = device)
  Y = X$matmul(beta) + torch_randn(n, 1, device = device) * 0.1^2

  net = make_network(p, latent, n_layers)
  net$to(device = device)

  opt_class = switch(optimizer,
    "ignite_adamw" = optim_ignite_adamw,
    "adamw" = optim_adamw,
    "sgd" = optim_sgd,
    "ignite_sgd" = optim_ignite_sgd
  )


  loss_fn = nn_mse_loss()
  net_parameters = net$parameters
  if (jit) {
    net = if (mlr3torch) {
      jit_trace(net, torch_randn(1, p, device = device), respect_mode = TRUE)
    } else {
      jit_trace(net, torch_randn(1, p, device = device), respect_mode = TRUE)$trainforward
    }
  }

  steps = ceiling(n / batch_size)

  get_batch = function(step, X, Y, batch_size) {
    list(
      x = X[seq((step - 1) * batch_size + 1, min(step * batch_size, n)), , drop = FALSE],
      y = Y[seq((step - 1) * batch_size + 1, min(step * batch_size, n)), , drop = FALSE]
    )
  }

  # this function should train the network for the given number of epochs and return the final training loss
  train_run = if (!mlr3torch) {
    do_step = function(input, target, opt) {
      opt$zero_grad()
      loss = loss_fn(net(input), target)
      loss$backward()
      opt$step()
    }


    function(epochs) {
      opt = opt_class(net_parameters, lr = lr)
      for (epoch in seq(epochs)) {
        for (step in seq_len(steps)) {
          batch = get_batch(step, X, Y, batch_size)
          do_step(batch$x, batch$y, opt)
        }
      }
    }

  } else {
    learner = LearnerTorchModel$new(
      task_type = "regr",
      optimizer = as_torch_optimizer(opt_class),
      ingress_tokens = list(x = ingress_ltnsr()),
      loss = as_torch_loss(nn_mse_loss)
    )
    learner$param_set$set_values(
      opt.lr = lr,
      device = device,
      drop_last = FALSE,
      jit_trace = FALSE,
      batch_size = batch_size,
      shuffle = FALSE,
      tensor_dataset = "device"
    )

    task = as_task_regr(data.table(
      x = as_lazy_tensor(X),
      y = as.numeric(Y)
    ), target = "y")

    function(epochs) {
      learner$.__enclos_env__$private$.network_stored = net
      learner$configure(epochs = epochs)
      learner$train(task)
    }
    
  }

  eval_run = function() {
      #net$eval()
      mean_loss = 0
      with_no_grad({
        for (step in seq_len(steps)) {
          batch = get_batch(step, X, Y, batch_size)
          y_hat = net(batch$x)
          loss = loss_fn(y_hat, batch$y)
          mean_loss = mean_loss + loss$item()
        }
      })
      mean_loss / steps
  }
  # warmup
  train_run(5)

  cuda_synchronize()
  #gc.time(TRUE)
  t0 = Sys.time()
  train_run(epochs)
  cuda_synchronize()
  #gc_time = gc.time()[3]
  t = as.numeric(difftime(Sys.time(), t0, units = "secs"))

  stats = cuda_memory_stats()
  memory = stats$reserved_bytes$all$current

  list(time = t, loss = eval_run(), memory = memory)
}
