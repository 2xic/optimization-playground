"""
https://facebookresearch.github.io/spdl/main/migration/pytorch.html
https://ai.meta.com/blog/spdl-faster-ai-model-training-with-thread-based-data-loading-reality-labs/?utm_source=twitter

TODO: test this, I had cmake install issues.
"""

from spdl.dataloader import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .add_source(range(len(dataset)))
    .pipe(
        dataset.__getitem__,
        concurrency=num_workers,
        output_order="input",
    )
    .add_sink(prefetch_factor)
    .build(num_threads=num_workers)
)

with pipeline.auto_stop():
    for batch, classes in dataloader:
        pass
