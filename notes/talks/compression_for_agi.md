# [Compression for AGI - Jack Rae](https://www.youtube.com/watch?v=dO4TPJkeaaU)

### Minimum description length
We want the kind of understanding that generalizes. 

Occams Razor etc.

----

[Chinese room thought experiment](https://en.wikipedia.org/wiki/Chinese_room#Chinese_room_thought_experiment) if a computer program translates from english to a chinese via series of rules, does it understand translation ? 

-> Larger representation = Lesser understanding 
-> Smaller representation = Better understanding 
-> MDL = Best version of the descriptor

$$|D|  -log P_f(D) + |f|$$
-> D = Lossless compression of the dataset
-> P_f -> log likelihood from a generative model f over D
-> |f| size of F

Compression is an objective we can't cheat. 

### Lossless compressions with LLMs
- Compression `D` amount of better next-token preidction
- Sum of next-token loss + code to innate and train model
  - But they ignore the parameter size ? -> WHat ? 
  - He says that because you can use arithmetic coding you don't need it. Not sure if I buy this. Need to think about this.

### Limitations
Lots of useful information in the world is not observable
- AlphaZero -> I.e self-play like insights is not something you would learn from just compression

## Conclusions
LM modelling advances are synonymous to greater compressions
- n-gram
- Longer context RNN models
- Transformers

Scale is not all you need. 

## Q&A
- Hutter prize might be missing something important with that scale and compute matters for improved models
  - Which se can see from recent progress, but I guess the point of the Hutter prize also is more about doing more with less data.
  - 
