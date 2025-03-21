library(torch)
library(here)

latent = 20000
layers = unlist(lapply(1:10, function(i) list(nn_linear(latent, latent), nn_relu())))
device = "cuda"

x = torch_randn(32, latent, device = device)

layers = c(layers, nn_linear(latent, 1))

net = do.call(nn_sequential, args = layers)
net = net$to(device = device)
net2 = net$clone(deep = TRUE)

net_jit = jit_trace(net, x, respect_mode = TRUE)$.__enclos_env__$private$find_method("trainforward")

res = bench::mark(
  other = {net(x); cuda_synchronize()},
  jit = {net_jit(x); cuda_synchronize()}
)
print(res)

