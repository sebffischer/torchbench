library(batchtools)
library(mlr3misc)

if (dir.exists("~/torchbenchmark")) {
  unlink("~/torchbenchmark", recursive = TRUE)
}

reg = makeExperimentRegistry(
  file.dir = "~/torchbenchmark",
  packages = c("checkmate", "reticulate")
)

# this defines the time_pytorch function
source("~/torchbench/time_rtorch.R")

batchExport(list(
  time_rtorch = time_rtorch
))


# The algorithm should return the total runtime needed for training, the SD, but also the performance of the training losses so we know it is all working
addProblem("runtime_train",
  data = NULL,
  fun = function(epochs, batch_size, n_layers, latent, n, p, optimizer, device,  cuda_allocator_reserved_rate, ...) {
    problem = list(
      epochs = assert_int(epochs),
      batch_size = assert_int(batch_size),
      n_layers = assert_int(n_layers),
      latent = assert_int(latent),
      n = assert_int(n),
      p = assert_int(p),
      optimizer = assert_choice(optimizer, c("adamw")),
      device = assert_choice(device, c("cuda", "cpu"))
    )

    problem
  }
)

# pytorch needs to be submitted with an active pytorch environment
# as I otherwise get the: OSError: libmkl_intel_lp64.so.2: cannot open shared object file: 
addAlgorithm("pytorch",
  fun = function(instance, job, data, ...) {
    f = function(...) {
      reticulate::use_condaenv("mlr3torch", required = TRUE)
      reticulate::source_python("~/torchbench/time_pytorch.py")
      time_pytorch(...)
    }
    #do.call(f, args = c(instance, list(seed = job$seed)))
    callr::r(f, args = c(instance, list(seed = job$seed)))
  }
)

# type can be: "base", "torchoptx", "torchoptx_jit", "ignite", and "mlr3torch_ignite" and indicates how the model is fit.
addAlgorithm("rtorch",
  fun = function(instance, job, type, ...) {
    assert_choice(type, c("jit", "standard"))
    callr::r(time_rtorch, args = c(instance, list(seed = job$seed, type = type)))
  }
)


problem_design = expand.grid(list(
  n          = 2000L,
  p          = 1000L,
  optimizer = c("adamw"),
  # training parameters
  epochs = 5L,
  latent = 500L,
  batch_size = 32L,
  device     = "cuda",
  n_layers = c(8L, 16L, 32L)
), stringsAsFactors = FALSE)

addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(type = c("standard", "jit")),
    pytorch = data.frame()
  ),
  repls = 2
)

if (FALSE) {
  # the ith repetition runs on the ith gpu, so we chunk all repetitions so they 
  # run sequentially on the same gpu
  jts = unwrap(getJobTable())[!is.na(type), ]
  tbl = data.frame(
    job.id = jts$job.id,
    chunk = jts$repl
  )
  tbl = tbl[1:8, ]
  submitJobs(tbl)
}

#submitJobs(ids)
#jt = getJobTable(ids) |> unwrap()
#jt$runtime = lapply(jt$job.id, function(x) loadResult(x)$time)

#library(data.table)
#jt = getJobTable() |> unwrap()
#tbl = rbindlist(lapply(findDone()[[1]], loadResult))
#tbl$type = jt$type

get_times = function() {
  lapply(findDone()[[1]], function(i) loadResult(i)$time)
}

get_losses = function() {
  lapply(findDone()[[1]], function(i) loadResult(i)$losses)
}

