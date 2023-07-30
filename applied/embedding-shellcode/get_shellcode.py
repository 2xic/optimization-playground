from pwnlib import shellcraft, asm

def get_shellcode():
    for i in dir(shellcraft.linux):
        if "_" in i:
            continue
        function_call = getattr(shellcraft.linux, i)
        try:
            clean_assembly = asm.disasm(asm.asm(function_call()), byte=False)
            yield clean_assembly, i
        except Exception as e:
            #print(e)
            pass
