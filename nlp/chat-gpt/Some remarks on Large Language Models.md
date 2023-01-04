[https://gist.github.com/yoavg/59d174608e92e845c8994ac2e234c8a9?utm_source=pocket_reader](Some remarks on Large Language Models)

The author starts with laying out his view on language models, and that before chatgpt was released he believed that training a big language model does not solving natural language understating, he also believes solving natural language understanding is AI-complete. One good argument he has for this is that to understand a text, you need to be able to "understand", "visualize" and "feel" it.

Later in the text he does mention that he find chatgpt impressive (but that there still is a way to go for lms), and that the view he had before was rooted based on older tech (before the transformer). He also believes that there was a big shift between 60B and 175 B parameters.

The author argues that the way the old LM models are trained does not ground them in reality. Knowing the position of the word blue, does not make the LM understand the color blue. In other words, it's just symbols, and the model can not create any relation to the symbols.
**However, when the model are given instructions during training (like summarize text) then it is able to create a relation to those instructions. The same is true for the models trained on code, having comments be the grounding for the code written. In addition to have RL with human feedback to be another grounding in how to have a dialogue.**


