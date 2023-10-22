"""
Playing around with the huggingface libraries

https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__.text_target
https://huggingface.co/docs/transformers/tasks/translation
"""
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
from accelerate import Accelerator
import torch
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

books = load_dataset("opus_books", "en-fr")

accelerator = Accelerator()
device = accelerator.device

"""
GPT does not fit on the machine I'm testing on :(
"""
#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)


checkpoint = "t5-small"
#checkpoint = "./my_awesome_opus_books_model/checkpoint-6000/"

"""
    >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
    >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    >>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
"""
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


optimizer = torch.optim.Adam(model.parameters())

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, padding=True)
    return model_inputs

books = books["train"].train_test_split(test_size=0.2)
tokenized_books = books.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_opus_books_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3,
    gradient_checkpointing=True,
    predict_with_generate=False,
    fp16=True,
    push_to_hub=False,
)
trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    args=training_args,
)

trainer.train()
#trainer.save_model("my_awesome_opus_books_model")

text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
translator = pipeline("translation", model=model.cpu(), tokenizer=tokenizer)
print(translator(text))
# [{'translation_text': 'Legumes partagent les ressources avec les bactéries fixatrices de azote.'}]
# [{'translation_text': 'Legumes teilen Ressourcen mit Stickstoff-fixierenden Bakterien.'}]
# Cool cool :=)
# 
"""
{'eval_loss': 0.21352511644363403, 'eval_runtime': 44.8822, 'eval_samples_per_second': 566.305, 'eval_steps_per_second': 17.713, 'epoch': 3.0}
"""
# LOL, checking the translation with google it is tally off though
# https://translate.google.com/?sl=de&tl=en&text=Legumes%20teilen%20Ressourcen%20mit%20Stickstoff-fixierenden%20Bakterien.&op=translate
# Outputs
# Legumes share resources with nitrogen-fixing bacteria. 
# Which is a the same input, but from german lol
# https://translate.google.com/?sl=fr&tl=en&text=Legumes%20teilen%20Ressourcen%20mit%20Stickstoff-fixierenden%20Bakterien.&op=translate
# Vegetables are prepared using Stickstoff-fixierenden Bakterien.
# which is weird. 
# 
# https://translate.google.com/?sl=en&tl=fr&text=legumes%20share%20resources%20with%20nitrogen-fixing%20bacteria.&op=translate
# les légumineuses partagent des ressources avec les bactéries fixatrices d’azote. 
# ^ should be the expected output
# 