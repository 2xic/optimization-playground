## Recourses
- [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)


## [Prompt-Engineering-Lecture-Elvis](https://raw.githubusercontent.com/dair-ai/Prompt-Engineering-Guide/main/lecture/Prompt-Engineering-Lecture-Elvis.pdf)
- Prompts should follow a basic structure, something like
    - Input
    - Context
    - Solution
- Various methods of doing prompts
    - Few shots
        - Give the model a few examples given the structure above, and it should be able to learn the patterns, and what to do
    - Chain of thought with few shot
        - By giving the input / context in a way like breaks down the task into smaller chains, it can help the model better understand the task.
        - See example slide 23
    - Zero-shot chain of thought
        - By just adding "let's think about this step by step" you will be able to get more correct answer, and possible without giving the model more examples
    - Self-consistency  
        - Generate multiple outputs to a query, and look at the output of the model that has the most consistent answer
    - [Program aided language model](https://arxiv.org/pdf/2211.10435.pdf)
        
## [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
References [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/pdf/2102.09690.pdf) that investigated few-shots in LLMs and GPT-3 more specially. For instance recency bias for the last label it saw, and that they use more common tokens instead of rare tokens. 
That paper is interesting, what they do is calibrate the model by feeding the model "dummy text" and seeing how the model responds. 



