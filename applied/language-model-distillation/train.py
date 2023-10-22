"""
https://huggingface.co/transformers/v4.4.2/custom_datasets.html


I get nan loss
- https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139
- https://huggingface.co/google/flan-t5-base/discussions/14
^ it is because of float16
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from dataset import generate_some_text
import time

device = "cuda"

class Teacher:
    def __init__(self) -> None:
        checkpoint = "bigcode/santacoder"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    def generate(self, text):
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(device)
        outputs = self.model.generate(inputs, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens=200)
        return self.tokenizer.decode(outputs[0])

class Student:
    def __init__(self) -> None:
        """
        TODO: Likely need to switch up the tokenizer
        """
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto", torch_dtype=torch.float32) # ^ NEED TO TRAIN WITH FLOAT32 UNLESS YOU WANT USLESS NAN 

       # self.model = T5ForConditionalGeneration(T5Config())#.from_pretrained("google/flan-t5-small", device_map="auto", torch_dtype=torch.float16)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.to(device)
        self.model.train() 

    def generate(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(device)
        outputs = self.model.generate(input_ids, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens=200)
        return self.tokenizer.decode(outputs[0])

    def predict(self, text, labels):
        self.optimizer.zero_grad()
        tokenized = self.tokenizer(text, text_target=labels, return_tensors="pt", max_length=256, padding=True)
        outputs = self.model(
            input_ids=tokenized.input_ids.to(device),
            labels=tokenized.labels.to(device),
            attention_mask=None
        )
        loss = outputs[0]
        #loss.backward()
        #optim.step()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
class Distillation:
    def __init__(self) -> None:
        self.teacher = Teacher()
        self.student = Student()

    def print_model(self, name, output):
        print("=" * 32)
        print(f"======={name}=======")
        print(output)
        print("")

    def train(self, text):
        teacher_output = self.teacher.generate(text)
        student_output = self.student.generate(text)
        for _ in range(10):
            loss = self.student.predict(text, labels=teacher_output)
            print(loss)
#        print("Before training", student_output)
 #       print("After training", self.student.generate(text))

if __name__ == "__main__":
    text = [
    #    "def print_hello_world():",
      #  "def start_server():",
      #  "def is_website_reachable(website):",
    ]
    distillation = Distillation()
    before = distillation.student.generate(
        "def is_website_reachable(website):"
    )    
    index = 0
    start = time.time()
    for i in generate_some_text():
        distillation.train(i)
        if 1_000 < index:
            break
        elif 5 < (time.time() - start) / 60:
            break
        index += 1
        print(index)
    after = distillation.student.generate(
        "def is_website_reachable(website):"
    )    
    print("before")
    print(before)
    print("after")
    print(after)

