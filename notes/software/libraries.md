## [https://github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)
- Sequence modeling toolkit - speech to text, summary, etc
- Implement a bunch of papers 

## [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- this is python 2 ... ? 
  - vqaEval.py
- So many dataset parsers
  - gqa
  - flickr
  - coco
- Not actually from scratch uses `transformers` :'(

## ollama
- Provided a simple away to get LLM running locally.
- https://ollama.com/blog/structured-outputs
- 

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama run llama2
```

It can probably be hosted somewhere and then you just curl into it. There also is a vast.ai docker image that uses https://github.com/open-webui/open-webui

