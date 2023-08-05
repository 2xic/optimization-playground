from get_shellcode import ShellcodeDataset
import json
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
import torch
import os
import math
from tokenizer import SimpleTokenizer
from get_debug_dataset import ShellcodeDebugDataset

class TokenizedShellCode(SimpleTokenizer):
    def __init__(self, dataloader) -> None:
        super().__init__()
        self.program_mapping = {}
        self.program_array = []
        self.raw_program = []
        self.token_program = []
        """
        How big sequence do you want to have as the input ? 
        """
        self.sequence_size = 256
        self.longest_program_length = 0
        self.dataloader = dataloader

    @property
    def n_tokens(self):
        return len(self.tokens_tracker.token_index) + 1

    def create(self):
        for output_format in self.dataloader.get_shellcode():
            sequence_program = []
            for j in output_format.code.split("\n"):
                instruction = self.dataloader.get_token_transformed(self, j)
                sequence_program += instruction
            self.raw_program.append(output_format.code)
            self.program_mapping[output_format.id] = sequence_program
            self.token_program.append("\n".join(list(map(str, sequence_program))))
            self.program_array.append(sequence_program)
            self.longest_program_length = max(len(sequence_program), self.longest_program_length)
        return self

    def save(self):
        with open("metadata.json", "w") as file:
            json.dump({
                "program_mapping":self.program_mapping,
                "token_index": self.tokens_tracker.token_index,
                "raw_program": self.raw_program,
                "token_program": self.token_program,
            }, file)
    
    def load(self):
        if os.path.isfile("metadata.json"):
            with open("metadata.json", "r") as file:
                data = json.load(file)
                self.program_mapping = data["program_mapping"]
                for key in data["token_index"]:
                    self.tokens_tracker.add_token(key)
                self.program_array = list(self.program_mapping.values())
                self.raw_program = data["raw_program"]
                self.token_program = data["token_program"]
            return self
        else:
            print("File does not exists, creating it")
            self.create()
            self.save()
            return self
            
    @property
    def program_tensor(self):
        self.tensor = None
        self.index_tensor = []
        for program_index, i in enumerate(self.program_array):
            if program_index % 10 == 0:
                print(f"program_index: {program_index}")
            padded_tensor = i + [self._PADDING] * (self.round_to_closest_sequence_batch_size(len(i)) - len(i))
            for index in range(0, len(padded_tensor), self.sequence_size):
                current = torch.tensor(padded_tensor[index:index+self.sequence_size]).unsqueeze(0)
                if self.tensor is None:
                    self.tensor = current
                else:
                    self.tensor = torch.concat(
                        (self.tensor, current)
                    )
                self.index_tensor.append(program_index)
        self.index_tensor = torch.tensor(self.index_tensor).long()
        return self.tensor.long()
    

    def round_to_closest_sequence_batch_size(self, x):
        base = self.sequence_size
        return base * math.ceil(x/base)

def get_current_dataset():
    return TokenizedShellCode(
        ShellcodeDataset()
        #ShellcodeDebugDataset()
    ).load()

def get_dataloader():
    dataset =  get_current_dataset()
    return get_raw_dataloader(dataset.program_tensor, dataset.index_tensor, batch_size=8), dataset

if __name__ == "__main__":
    TokenizedShellCode().create().save()
