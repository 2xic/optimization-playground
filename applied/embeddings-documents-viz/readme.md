How correlated are Claude and OpenAI embeddings ? And even more important, how does our embeddings fit into it?

![example](./example.png)

## How to use 
Define the connection information inside `build_data_file.py` and then run it.

```bash
LOCAL_HOST="" python3 build_data_file.py
python3 server.py --file embeddings.json
```
