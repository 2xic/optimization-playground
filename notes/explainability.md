## [Language models can explain neurons in language models](https://openai.com/research/language-models-can-explain-neurons-in-language-models)
TLDR: OpenAI uses GPT-4 to describe the neurons in GPT-2.
In addition they rerelease a dataset with the (imperfect) description of all the neurons in GPT-2

The technique
- Show neuron activity corresponding with some text, and have GPT-4 explain it
- Then GPT-4 will give an estimate for which token in the text the neuron will fire on
- Compare the GPT-4 predicted activation with the actual activation = score

Some things the authors found out 
- Giving GPT-4 the option to create iterations of the explanations. For instance by having the model generate counterexamples and revise the explanation

[@UpolEhsan](https://twitter.com/upolehsan/status/1659562051541381120?s=12&utm_source=pocket_saves) had some "critique" for instance that "descriptions are not explanations", and same is true for visualizations. 

[Hackernews threads](https://news.ycombinator.com/item?id=35877402&p=2) is funny.

[Eliezer view + gdb](https://twitter.com/gdb/status/1656375845018271751)

