from tokenizer import SimpleTokenizer
from pwnlib import shellcraft, asm
from output_format import OutputFormat
"""
Simple dataloader 
"""
class ShellcodeDataset:
    def get_shellcode(self):
        for name in dir(shellcraft.linux):
            if "_" in name:
                continue
            function_call = getattr(shellcraft.linux, name)
            try:
                clean_assembly = asm.disasm(asm.asm(function_call()), byte=False)
                yield OutputFormat(id=name, name=name, code=clean_assembly)
            except Exception as e:
                pass

    def get_token_transformed(self, tokenizer: SimpleTokenizer, instruction):
        split = instruction.split(":")
        offset = split[0].strip()
        instruction = split[1]

        instruction_split = instruction.strip().split("   ")
        opcode = instruction_split[0]
        arguments = list(map(
            lambda x: tokenizer.tokens_tracker.add_token(x.strip()),
            instruction_split[1].strip().split(",")
        )) if len(instruction_split) == 2 else []

        instruction = [
            tokenizer.start_offset,
            tokenizer.tokens_tracker.add_token(offset),
            tokenizer.end_offset,
            tokenizer.start_instruction,
            tokenizer.tokens_tracker.add_token(opcode),
        ] + arguments + [tokenizer.end_instruction]

        return instruction
