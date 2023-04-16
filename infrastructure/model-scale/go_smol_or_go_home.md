### [Go smol or go home](http://web.archive.org/web/20230410155657/https://www.harmdevries.com/post/model-size-vs-compute-overhead/)
[Original link](https://www.harmdevries.com/post/model-size-vs-compute-overhead/)
[Tweet thread](https://twitter.com/harmdevries77/status/1646524056538316805?utm_source=pocket_reader)
- In most cases you should not train a compute optimal LLM, but instead spent extra compute on gaining a smaller model. This makes the model lighter to run and allows for more innovation.
- One little known fact is that `Chinchilla scaling laws` can be used to also determine how much more compute you need to make the model smaller.
  - Requirers some reworking of the math function layed out in the blogpost
- The compute overhead scales up faster and faster the smaller you want your model
    - "50% of the compute-optimal model leads to 20% compute overhead" (quote from tweet thread)
    - "30% results in a 100% overhead"  (quote from tweet thread)
- 