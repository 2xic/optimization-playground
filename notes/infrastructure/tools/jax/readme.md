## [Jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)

## initial experience
Running the first example I get this 
```
jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.
```
Ugh, but the only change needed was to go from int to float and things work.

## Example working
The jax code looks a lot cleaner actually, this might be very useful for more optimization (non nn layers) work.

## Neural networks (?)
- [They don't really have the layer abstractions](https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html)
  - They got some opcodes though, but no layers [https://jax.readthedocs.io/en/latest/jax.nn.html](https://jax.readthedocs.io/en/latest/jax.nn.html) and [https://jax.readthedocs.io/en/latest/jax.example_libraries.stax.html](https://jax.readthedocs.io/en/latest/jax.example_libraries.stax.html)
