"""
https://huggingface.co/transformers/v4.4.2/custom_datasets.html


I get nan loss
- https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139
- https://huggingface.co/google/flan-t5-base/discussions/14
^ it is because of float16

https://huggingface.co/transformers/v3.3.1/training.html
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from dataset import generate_some_text
import time

debug = False
device = "cuda" if not debug else "cpu"
token_length = 512

class Teacher:
    def __init__(self) -> None:
        checkpoint = "bigcode/santacoder"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    def generate(self, text):
        #inputs = self.tokenizer.encode(text, return_tensors="pt").to(device)
        tokenized = self.tokenizer(text, return_tensors="pt")#
        attention_mask = tokenized['attention_mask']

        outputs = self.model.generate(
            input_ids=tokenized.input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=token_length,
        )
        return self.tokenizer.decode(outputs[0])

class Student:
    def __init__(self) -> None:
        """
        TODO: Likely need to switch up the tokenizer
        """
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto", torch_dtype=torch.float32) # ^ NEED TO TRAIN WITH FLOAT32 UNLESS YOU WANT USLESS NAN 

        # we disable to optimize the model - yes yes yes
        for param in list(self.model.encoder.parameters()) + list(self.model.shared.parameters()):
            param.requires_grad = False
            
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)
        self.model.to(device)
        self.model.train() 

    def generate(self, text):
        with torch.no_grad():
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(device)
            outputs = self.model.generate(input_ids, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens=token_length)
            return self.tokenizer.decode(outputs[0])

    def predict(self, text, labels):
        tokenized = self.tokenizer(text, text_target=labels, return_tensors="pt", padding=True)
        attention_mask = tokenized['attention_mask']
        outputs = self.model(
            input_ids=tokenized.input_ids.to(device),
            labels=tokenized.labels.to(device),
            attention_mask=attention_mask.to(device)
        )
        loss = outputs[0]
        return loss
    
class Distillation:
    def __init__(self) -> None:
        self.teacher = Teacher()
        self.student = Student()
        self.index = 0
        self.total_loss = 0

    def print_model(self, name, output):
        print("=" * 32)
        print(f"======={name}=======")
        print(output)
        print("")

    def single_step_train(self, text):
        teacher_output = self.teacher.generate(text)
        loss = self.student.predict(text, labels=teacher_output)
        loss.backward()
        self.index += 1

        self.total_loss += loss.item()

        if self.index % 8 == 0:
            self.student.optimizer.step()
            self.student.optimizer.zero_grad()
            print(self.total_loss)
            self.total_loss = 0

    def single_step_train_hardcoded(self):
        text = "hello"
        teacher_output = "hello world, this is some text that the model should learn to output now!"
        loss = self.student.predict(text, labels=teacher_output)
        loss.backward()
        self.index += 1

        self.total_loss += loss.item()

        if self.index % 8 == 0:
            self.student.optimizer.step()
            self.student.optimizer.zero_grad()
            print(self.total_loss)
            self.total_loss = 0

if __name__ == "__main__":
    distillation = Distillation()
    index = 0
    start = time.time()
    train_on_dataset = True

    if train_on_dataset:
        before = distillation.student.generate(
            "def "
        )
        for i in generate_some_text():
            distillation.single_step_train(i)
            if 1_000 < index:
                print("1000 indexes")
                break
            elif 5 < (time.time() - start) / 60:
                print("5 minute timeout")
                break
            index += 1
            print(index)
            after = distillation.student.generate(
                "def "
            )
            print("before")
            print(before)
            print("after")
            print(after)
    else:
        before = distillation.student.generate(
            "hello"
        )
        for i in range(1_000):
            distillation.single_step_train_hardcoded()
            if 1_000 < index:
                print("1000 indexes")
                break
            elif 5 < (time.time() - start) / 60:
                print("5 minute timeout")
                break
            index += 1
            print(index)
            after = distillation.student.generate(
                "hello"
            )
            print("before")
            print(before)
            print("after")
            print(after)
