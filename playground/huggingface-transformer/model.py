"""
https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__.text_target
https://huggingface.co/docs/transformers/tasks/translation
"""
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
from accelerate import Accelerator
import torch

books = load_dataset("opus_books", "en-fr")

accelerator = Accelerator()
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained("gpt2").add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

optimizer = torch.optim.Adam(model.parameters())

books = books["train"].train_test_split(test_size=0.2)

for i in books["train"]:
    row = i["translation"]
    en = row["en"]
    fr = row["fr"]
    print((en, fr))

    inputs = tokenizer(en, text_target=fr, return_tensors="pt", padding=True).to(device)
 #   print(expected_output)
  #  print(expected_output.shape)
    outputs = model(**inputs)

    loss = outputs.loss
    print(loss)
