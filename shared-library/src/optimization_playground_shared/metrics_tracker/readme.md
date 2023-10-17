## html and some plots is all you need (for experiments tracking)
Very simple metrics tracker to make it easier to keep track of current training round and older runs.
You also get a code diff so you can see what actually changed between experiment runs.

## Todo
- I want a way to launch `n` runs with the same configs to make sure there is no issue with the local minima or similar. 

# Other tools out there

## https://github.com/IDSIA/sacred
- Nice idea for reproducibility with the decorators

## why not use w&b ? 
I have used this before, but I think it makes sense to just have something self-hosted and it has no code diff.
