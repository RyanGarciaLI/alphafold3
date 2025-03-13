import jax
import jax.numpy as jnp

enable_vis = True

def print_tensor(marker, tensor):
    dtype = tensor.dtype
    # if tensor.dtype == paddle.bfloat16:
    #     tensor = tensor.cast(paddle.float32)
    # mean = jnp.mean(tensor)
    median = jnp.median(tensor)
    std = jnp.std(tensor)
    max_v = jnp.max(tensor)
    # min_v = jnp.min(tensor)
    jax.debug.print(">>>>>> [DEBUG] {} max {} median {} std {} dtype {} shape {}", 
                    marker, max_v, median, std, tensor.dtype, tensor.shape)