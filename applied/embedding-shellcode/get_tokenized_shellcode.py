from get_shellcode import get_shellcode
import json
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
import torch
import os
import math

class TokenTracker:
    def __init__(self) -> None:
        self.token_index = {}
        self.index_token = {}

    def add_token(self, token):
        if token not in self.token_index:
            index = len(self.token_index)
            self.token_index[token] = index
            self.index_token[index] = token
            return index
        else:
            return self.token_index[token]
        
class TokenizedShellCode:
    def __init__(self) -> None:
        self.tokens_tracker = TokenTracker()
        self._PADDING = self.tokens_tracker.add_token("<PADDING>")
        self.start_offset = self.tokens_tracker.add_token("<START_OFFSET>")
        self.end_offset = self.tokens_tracker.add_token("<END_OFFSET>")
        self.start_instruction = self.tokens_tracker.add_token("<START_INSTRUCTION>")
        self.end_instruction = self.tokens_tracker.add_token("<END_T_INSTRUCTION>")
        self.program_mapping = {}
        self.program_array = []
        self.raw_program = []
        self.token_program = []
        self.sequence_size = 100 # 356
        self.longest_program_length = 0

    @property
    def n_tokens(self):
        return len(self.tokens_tracker.token_index) + 1

    def create(self):
        for (raw_program, name) in get_shellcode():
            program = []
            for j in raw_program.split("\n"):
                split = j.split(":")
                offset = split[0].strip()
                instruction = split[1]

                instruction_split = instruction.strip().split("   ")
                opcode = instruction_split[0]
                arguments = list(map(
                    lambda x: self.tokens_tracker.add_token(x.strip()),
                    instruction_split[1].strip().split(",")
                )) if len(instruction_split) == 2 else []

                instruction = [
                    self.start_offset,
                    self.tokens_tracker.add_token(offset),
                    self.end_offset,
                    self.start_instruction,
                    self.tokens_tracker.add_token(opcode),
                ] + arguments + [self.end_instruction]
                program += instruction
            self.raw_program.append(raw_program)
            self.program_mapping[name] = program
            self.token_program.append("\n".join(
                list(map(str, program))
            ))
            self.program_array.append(program)
            self.longest_program_length = max(len(program), self.longest_program_length)
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
             #   print(current.shape)
             #   print(len(padded_tensor), len(i))
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

def get_dataloader():
    dataset = TokenizedShellCode().load()
    return get_raw_dataloader(dataset.program_tensor, dataset.index_tensor, batch_size=8), dataset

if __name__ == "__main__":
    TokenizedShellCode().create().save()
    
