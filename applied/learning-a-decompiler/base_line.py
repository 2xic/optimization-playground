# we use this nice tutorial from huggingface - hopefully it works https://huggingface.co/learn/nlp-course/chapter7/4?fw=tf
# https://medium.com/@anyuanay/fine-tuning-the-pre-trained-t5-small-model-in-hugging-face-for-text-summarization-3d48eb3c4360
# https://huggingface.co/docs/transformers/en/model_doc/t5 are all too large for the machine I'm renting.
# https://huggingface.co/docs/transformers/v4.18.0/en/performance


from transformers import AutoTokenizer
from dataset_loader import dataset
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model_checkpoint = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
bin_sentence = dataset["train"][1]["bin"]
source_sentence = dataset["train"][1]["source"]

inputs = tokenizer(bin_sentence, text_target=source_sentence)


def preprocess_function(examples):
   # print(examples)
    inputs = examples["bin"]# for ex in examples]
    targets = examples["source"] #for ex in examples]
    model_inputs = tokenizer(inputs, text_target=targets)
    return model_inputs

tokenized_datasets = dataset["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
#print(tokenized_datasets[0])
#print(tokenized_datasets[0][1].shape)
#exit(0)q

training_args = Seq2SeqTrainingArguments(
    output_dir="my_fine_tuned_t5_small_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adafactor",
    fp16=True,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)
trainer.train()
