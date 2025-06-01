"""
Just a opcode mapper
"""

from typing import Dict
from copy import deepcopy


class Opcode:
    def __init__(self, name, size=1) -> None:
        self.name = name
        self.size = size
        self.args = ""

    def __str__(self) -> str:
        if len(self.args) > 0:
            return f"{self.name} {self.args}"
        return self.name

    def __repr__(self) -> str:
        return self.__str__()


class PushOpcode(Opcode):
    def __init__(self, name, size=1):
        super().__init__(name, size)


JUMP_DEST = 0x5B

opcodes_mapping: Dict[int, Opcode] = {
    0x00: Opcode("STOP"),
    0x01: Opcode("ADD"),
    0x02: Opcode("MUL"),
    0x03: Opcode("SUB"),
    0x04: Opcode("DIV"),
    0x05: Opcode("SDIV"),
    0x06: Opcode("MOD"),
    0x07: Opcode("SMOD"),
    0x08: Opcode("ADDMOD"),
    0x09: Opcode("MULMOD"),
    0x0A: Opcode("EXP"),
    0x0B: Opcode("SIGNEXTEND"),
    0x10: Opcode("LT"),
    0x11: Opcode("GT"),
    0x12: Opcode("SLT"),
    0x13: Opcode("SGT"),
    0x14: Opcode("EQ"),
    0x15: Opcode("ISZERO"),
    0x16: Opcode("AND"),
    0x17: Opcode("OR"),
    0x18: Opcode("XOR"),
    0x19: Opcode("NOT"),
    0x20: Opcode("SHA3"),
    0x30: Opcode("ADDRESS"),
    0x31: Opcode("BALANCE"),
    0x32: Opcode("ORIGIN"),
    0x33: Opcode("CALLER"),
    0x34: Opcode("CALLVALUE"),
    0x35: Opcode("CALLDATALOAD"),
    0x36: Opcode("CALLDATASIZE"),
    0x37: Opcode("CALLDATACOPY"),
    0x38: Opcode("CODESIZE"),
    0x39: Opcode("CODECOPY"),
    0x40: Opcode("BLOCKHASH"),
    0x41: Opcode("COINBASE"),
    0x42: Opcode("TIMESTAMP"),
    0x43: Opcode("NUMBER"),
    0x44: Opcode("DIFFICULTY"),
    0x45: Opcode("GASLIMIT"),
    0x46: Opcode("CHAINID"),
    0x47: Opcode("SELFBALANCE"),
    0x48: Opcode("BASEFEE"),
    0x50: Opcode("POP"),
    0x51: Opcode("MLOAD"),
    0x52: Opcode("MSTORE"),
    0x53: Opcode("MSTORE8"),
    0x54: Opcode("SLOAD"),
    0x55: Opcode("SSTORE"),
    0x56: Opcode("JUMP"),
    0x57: Opcode("JUMPI"),
    JUMP_DEST: Opcode("PC"),
    0x59: Opcode("MSIZE"),
    0x5F: Opcode("PUSH0"),
    0x80: Opcode("DUP1"),
    0x81: Opcode("DUP2"),
    0x82: Opcode("DUP3"),
    0x83: Opcode("DUP4"),
    0x84: Opcode("DUP5"),
    0x85: Opcode("DUP6"),
    0x86: Opcode("DUP7"),
    0x87: Opcode("DUP8"),
    0x88: Opcode("DUP9"),
    0x89: Opcode("DUP10"),
    0x90: Opcode("SWAP1"),
    0x91: Opcode("SWAP2"),
    0x92: Opcode("SWAP3"),
    0x93: Opcode("SWAP4"),
    0x94: Opcode("SWAP5"),
    0x95: Opcode("SWAP6"),
    0x96: Opcode("SWAP7"),
    0x97: Opcode("SWAP8"),
    0x98: Opcode("SWAP9"),
    0x99: Opcode("SWAP10"),
    0x00: Opcode("STOP"),
    0x01: Opcode("ADD"),
    0x02: Opcode("MUL"),
    0x03: Opcode("SUB"),
    0x04: Opcode("DIV"),
    0x05: Opcode("SDIV"),
    0x06: Opcode("MOD"),
    0x07: Opcode("SMOD"),
    0x08: Opcode("ADDMOD"),
    0x09: Opcode("MULMOD"),
    0x0A: Opcode("EXP"),
    0x0B: Opcode("SIGNEXTEND"),
    0x1A: Opcode("BYTE"),
    0x1B: Opcode("SHL"),
    0x1C: Opcode("SHR"),
    0x1D: Opcode("SAR"),
    0x3A: Opcode("GASPRICE"),
    0x3B: Opcode("EXTCODESIZE"),
    0x3C: Opcode("EXTCODECOPY"),
    0x3D: Opcode("RETURNDATASIZE"),
    0x3E: Opcode("RETURNDATACOPY"),
    0x3F: Opcode("EXTCODEHASH"),
    0x5A: Opcode("GAS"),
    0x5B: Opcode("JUMPDEST"),
    0x8A: Opcode("DUP11"),
    0x8B: Opcode("DUP12"),
    0x8C: Opcode("DUP13"),
    0x8D: Opcode("DUP14"),
    0x8E: Opcode("DUP15"),
    0x8F: Opcode("DUP16"),
    0x9A: Opcode("SWAP11"),
    0x9B: Opcode("SWAP12"),
    0x9C: Opcode("SWAP13"),
    0x9D: Opcode("SWAP14"),
    0x9E: Opcode("SWAP15"),
    0x9F: Opcode("SWAP16"),
    0xA0: Opcode("LOG0"),
    0xA1: Opcode("LOG1"),
    0xA2: Opcode("LOG2"),
    0xA3: Opcode("LOG3"),
    0xA4: Opcode("LOG4"),
    0xF0: Opcode("CREATE"),
    0xF1: Opcode("CALL"),
    0xF2: Opcode("CALLCODE"),
    0xF3: Opcode("RETURN"),
    0xF4: Opcode("DELEGATECALL"),
    0xF5: Opcode("CREATE2"),
    0xFA: Opcode("STATICCALL"),
    0xFD: Opcode("REVERT"),
    0xFE: Opcode("INVALID"),
    0xFF: Opcode("SELFDESTRUCT"),
    0x5C: Opcode("TLOAD"),
    0x5D: Opcode("TSTORE"),
    -1: Opcode("INVALID"),
}
for i in range(0, 32):
    opcodes_mapping[0x60 + i] = PushOpcode(f"PUSH{i + 1}", size=i + 2)


def get_opcodes_from_bytecode(stream):
    index = 0
    opcodes = []
    while index < len(stream):
        opcode_id = stream[index]
        opcode = deepcopy(opcodes_mapping.get(opcode_id, opcodes_mapping[-1]))

        index += 1
        for i in range(1, opcode.size):
            if index < len(stream):
                opcode.args += "{:02x}".format(stream[index])
                index += 1
        opcodes.append(opcode)
    return opcodes
