from utils.load_mode_from_checkpoint import iterate_over_dataset, load_model_from_path
import os
from utils.web_dataloader import WebDataloader
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
    argmax_sampling,
)
from dotenv import load_dotenv
import json
from utils.mixture_dataloader import WebDataloaderMixture

load_dotenv()

prompts = [
    "Hello!",
    "What is 1 + 1 ? ",
    "What is Python?",
    "Python is ",
    "<|user|>\nWhat is 1 + 1 ?\n<|end|>",
    "<|user|>What is 1 + 1 ?<|end|>",
    "<|user|>\n write a function to add two numbers\n<|end|>",
    "<|user|>\n write a function to add two numbers\n<|end|>\n<|assistant|>\n",
    "<|user|>\n Hello! \n<|end|>\n<|assistant|>\n",
    "<|user|>\n Hey! \n<|end|>\n",
    "<|user|>",
    "<|user|> Hi<|end|><|assistant|> ",
]
# Dataset we are trying to finetune on top of
dataloader_dataset = (
    "smoltalk-256+everyday-conversations-256+self-oss-instruct-sc2-H4-256"
)
# Raw pretrained dataset
# underlying_dataset = "medium-web-256-v2"
underlying_dataset = "smedium-web-256"

dataloader = dataloader = WebDataloader(
    os.environ["WEB_DATALOADER"],
    underlying_dataset,
    batch_size=1024,
)


def inspect_dataset():
    top_dataloader = WebDataloaderMixture(
        list(
            [
                WebDataloader(
                    os.environ["WEB_DATALOADER"],
                    dataset,
                    batch_size=1024,
                )
                for dataset in [
                    "smoltalk-256",
                    "everyday-conversations-256",
                    "self-oss-instruct-sc2-H4-256",
                ]
            ]
        )
    )
    for batch in top_dataloader:
        X, _ = batch["x_tokens"], batch["y_tokens"]
        a = X.tolist()[0]
        #    print(a)
        detokenzied = dataloader.detokenize(a)
        print(detokenzied)


# inspect_dataset()

# exit(0)


# TODO: figure out why this is included so weird ...
def remove_padding_sampling(logits, underling_func):
    logits[:, dataloader.padding_index] = float("-inf")
    return underling_func(logits)


end_token_id = dataloader.convert_token_to_id("<|end|>")

for base_model_path, stats in iterate_over_dataset(dataloader_dataset, max_age_days=10):
    if stats["steps"] < 907027:
        continue
    (model, model_config) = load_model_from_path(base_model_path)
    model_response = []
    for text in prompts:
        doc_tensors = dataloader.tokenize([text])[0]
        doc_tensors = doc_tensors[0]
        model_temperature_sampling = model.generate(
            doc_tensors, 128, temperature_sampling, end_token_id=end_token_id
        )
        model_argmax_sampling = model.generate(
            doc_tensors, 128, argmax_sampling, end_token_id=end_token_id
        )
        beam_search = model.beam_search(doc_tensors, 128, end_token_id=end_token_id)

        model_response.append(
            {
                "model_temperature_sampling": dataloader.detokenize(
                    model_temperature_sampling
                ),
                "model_argmax_sampling": dataloader.detokenize(model_argmax_sampling),
                "model_beam_search": dataloader.detokenize(beam_search),
            }
        )
    datapoint = {
        "base_model_path": base_model_path,
        "model_responses": model_response,
        "stats": stats,
        "parameters": {
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters()),
        },
    }
    os.makedirs(".temp", exist_ok=True)
    path = os.path.join(
        ".temp",
        os.path.basename(os.path.dirname(base_model_path))
        + os.path.basename(base_model_path)
        + ".json",
    )
    with open(path, "w") as file:
        file.write(json.dumps(datapoint, indent=4))

# print(json.dumps(model_response, indent=4))
