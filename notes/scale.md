## [Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/pdf/2010.14701)
- Cross-entropy scaling laws are seen in four domains
  - generative image modeling
  - multimodal image <> text
  - video modeling
  - mathematical problem solving
- All modalities also seem to scale the same away
- Transformers are great?

## [On the Predictability of Pruning Across Scales](https://arxiv.org/pdf/2006.10621)
- Interesting paper: can we predict the error rate of a pruned network ? 
  - They mostly study resnet like architectures
- See figure 7 and figure 8 for context

## [gzip Predicts Data-dependent Scaling Laws](https://arxiv.org/pdf/2405.16684)
[Twitter thread](https://x.com/khoomeik/status/1795477359933706272)

1. Scaling laws are sensitive to the complexity of the dataset
2. gzip is a good predictor of said complexity

## [inverse scaling prize](https://github.com/inverse-scaling/prize)
[Paper](https://arxiv.org/pdf/2306.09479)

- Many benchmark results will improve when the model is scaled up, but what kind of tasks do worse on a bigger model ? 

## [Extrapolating performance in language modeling benchmarks](https://epochai.org/files/llm-benchmark-extrapolation.pdf)
Did some evaluation and testing of scaling laws on LLMs. The research show that benchmark performance is predicable using scaling laws. 
Aggregated benchmark performance is more predicable than individual performance.

## [Scaling Laws for Transfer](https://arxiv.org/pdf/2102.01293)
How does scaling laws affect transfer learning? 
- "Ossification" - can a models performance be hurt by pre-training ? In certain scenarios, it might. Smaller models might benefits from being trained from scratch, but larger onces are less effected (Figure 5).
  - This also happens even if the models are trained for much longer, smaller fine-tuned models have no way of recovering.
- On smaller datasets pre-training is more compute efficient.
- They created a unified scaling law for pre-training that looks a lot similar to the existing onces.
- **Conclusion**: scaling laws also existing for pre-training.

## [DEEP LEARNING SCALING IS PREDICTABLE, EMPIRICALLY](https://arxiv.org/pdf/1712.00409)
The first paper (?) to notice power laws with model/data/computation scales.

## [An Empirical Exploration in Quality Filtering of Text Data](https://arxiv.org/pdf/2109.00698)
Aggressively filtering out data from common crawl results in a worse model. The author speculates the reason for this being that the proxy metrics hurts the performance of the actual metric.

## [INFERENCE SCALING LAWS: AN EMPIRICAL ANALYSIS OF COMPUTE-OPTIMAL INFERENCE FOR LLM PROBLEM-SOLVING](https://arxiv.org/pdf/2408.00724)
Smaller models can beat larger models when using novel search. Example Llemma-7B model beats Llemma-34B when equipped with a novel tree search algorithm.
- Instead of using MCTs they use `REBASE` (described in section 3.1.2)

## Models can learn A -> B, but not B -> A
- [Good tweet on it](https://x.com/cloneofsimo/status/1888975591359320273)
  - https://arxiv.org/pdf/2407.21783#page=20
  - https://arxiv.org/pdf/2309.12288

I should explore this more.
