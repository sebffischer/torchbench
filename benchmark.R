library(batchtools)
library(mlr3misc)

if (dir.exists("~/torchbenchmark1")) {
  unlink("~/torchbenchmark1", recursive = TRUE)
}

reg = makeExperimentRegistry(
  file.dir = "~/torchbenchmark1",
  packages = c("checkmate")
)

# this defines the time_pytorch function
source("~/torchbench/time_rtorch.R")

batchExport(list(
  time_rtorch = time_rtorch
))

# The algorithm should return the total runtime needed for training, the SD, but also the performance of the training losses so we know it is all working
addProblem("runtime_train",
  data = NULL,
  fun = function(epochs, batch_size, n_layers, latent, n, p, optimizer, device, ...) {
    problem = list(
      epochs = assert_int(epochs),
      batch_size = assert_int(batch_size),
      n_layers = assert_int(n_layers),
      latent = assert_int(latent),
      n = assert_int(n),
      p = assert_int(p),
      optimizer = assert_choice(optimizer, c("ignite_adamw", "adamw", "sgd", "ignite_sgd")),
      device = assert_choice(device, c("cuda", "cpu"))
    )

    problem
  }
)

# pytorch needs to be submitted with an active pytorch environment
addAlgorithm("pytorch",
  fun = function(instance, job, data, jit, ...) {
    f = function(...) {
      library(reticulate)
      x = try({
        reticulate::use_condaenv("mlr3torch", required = TRUE)
        reticulate::source_python("~/torchbench/time_pytorch.py")
        print(reticulate::py_config())
        time_pytorch(...)
      }, silent = TRUE)
      print(x)

    }
    args = c(instance, list(seed = job$seed, jit = jit))
    #do.call(f, args)
    callr::r(f, args = args)
  }
)

addAlgorithm("rtorch",
  fun = function(instance, job, opt_type, jit,...) {
    assert_choice(opt_type, c("standard", "ignite"))
    if (opt_type == "ignite") {
      instance$optimizer = paste0("ignite_", instance$optimizer)
    }
    #do.call(time_rtorch, args = c(instance, list(seed = job$seed, jit = jit)))
    callr::r(time_rtorch, args = c(instance, list(seed = job$seed, jit = jit)))
  }
)

addAlgorithm("mlr3torch",
  fun = function(instance, job, opt_type, jit, ...) {
    if (opt_type == "ignite") {
      instance$optimizer = paste0("ignite_", instance$optimizer)
    }
    #do.call(time_rtorch, args = c(instance, list(seed = job$seed, mlr3torch = TRUE, jit = jit)))
    callr::r(time_rtorch, args = c(instance, list(seed = job$seed, mlr3torch = TRUE, jit = jit)))
  }
)

# global config:
REPLS = 1L
EPOCHS = 1L
N = 2000L
P = 1000L

# cuda experiments:


problem_design = expand.grid(list(
  n          = N,
  p          = P,
  epochs = EPOCHS,
  latent = c(1000, 2500, 5000),
  optimizer = c("sgd", "adamw"),
  batch_size = 32L,
  device     = "cuda",
  n_layers = c(1L, 4L, 16L)
), stringsAsFactors = FALSE)


addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(
      jit = c(FALSE, TRUE),
      opt_type = c("ignite"),
      tag = "cuda_exp"
    ),
    mlr3torch = data.frame(
      jit = c(FALSE, TRUE),
      opt_type = c("ignite"),
      tag = "cuda_exp"
    ),
    pytorch = data.frame(
      jit = c(FALSE, TRUE),
      tag = "cuda_exp"
    )
  ),
  repls = REPLS
)

# cpu experiments:
# (need smaller networks, otherwise too expensive with the cuda config)

problem_design = expand.grid(list(
  n          = N,
  p          = P,
  epochs = EPOCHS,
  # factor 10 smaller than cuda
  latent = c(100, 250, 500),
  optimizer = c("sgd", "adamw"),
  batch_size = 32L,
  device     = "cpu",
  n_layers = c(1L, 4L, 16L)
), stringsAsFactors = FALSE)

addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(
      jit = c(FALSE, TRUE),
      opt_type = c("ignite"),
      tag = "cpu_exp"
    ),
    mlr3torch = data.frame(
      jit = c(FALSE, TRUE),
      opt_type = c("ignite"),
      tag = "cpu_exp"
    ),
    pytorch = data.frame(
      jit = c(FALSE, TRUE),
      tag = "cpu_exp"
    )
  ),
  repls = REPLS
)


# ignite vs non-ignite
# here we don't need to run so many experiments, just need to show that one is clearly faster

problem_design = expand.grid(list(
  n          = N,
  p          = P,
  epochs = EPOCHS,
  # factor 10 smaller than cuda
  latent = c(1000),
  optimizer = c("sgd", "adamw"),
  batch_size = 32L,
  device     = c("cpu", "cuda"),
  n_layers = 16L
), stringsAsFactors = FALSE)

addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(
      jit = FALSE,
      opt_type = c("standard", "ignite"),
      tag = "ignite_exp"
    ),
    mlr3torch = data.frame(
      jit = FALSE,
      opt_type = c("standard", "ignite"),
      tag = "ignite_exp"
    )
  ),
  repls = REPLS
)

get_result = function(ids, what) {
  if (is.null(ids)) ids = findDone()[[1]]
  sapply(ids, function(i) {
    res = loadResult(i)[[what]]
    if (is.null(res)) return(NA)
    res
  })
}

summarize = function(ids = NULL) {
  jt = getJobTable() |> unwrap()
  if (!is.null(ids)) jt = jt[ids, ]
  jt = jt[, c("n_layers", "jit", "optimizer", "batch_size", "device", "opt_type", "algorithm", "repl", "tag", "latent")]
  jt$time = get_result(ids, "time")
  jt$loss = get_result(ids, "loss")
  jt$memory = get_result(ids, "memory") / 2^30
  return(jt)
}
