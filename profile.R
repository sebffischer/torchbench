library(mlr3torch)

task = tsk("lazy_iris")

learner = lrn("classif.mlp", neurons = rep(500, 16), jit_trace = TRUE, batch_size = 32, shuffle = TRUE, tensor_dataset = TRUE, epochs = 300)

p = profvis::profvis({
    learner$train(task)
  } 
 ,simplify = FALSE)

htmlwidgets::saveWidget(p, here::here("profile.html"), selfcontained = TRUE)
