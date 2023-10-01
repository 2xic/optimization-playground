"""
One approach is to just repeat the same messages again and again
"""
from PIL import Image
import numpy as np
from collections import defaultdict


class Channel:
    def __init__(self):
        image = Image.open("cover.png").convert("RGB")
        self.source = np.asarray(image)
        self.target = np.zeros(self.source.shape).astype(np.uint8)
        self.p_flip = 0.1

    def send(self):
        print(self.source.shape)
        for x in range(self.source.shape[0]):
            for y in range(self.source.shape[1]):
                for z in range(self.source.shape[2]):
                    symbol = self.source[x][y][z]
                    yield (x, y, z, symbol)

    def transfer(self, symbol):
        if np.random.rand() <= self.p_flip:
            symbol = (symbol + np.random.rand() * 255) % 255
        return symbol

    def decode(self, x, y, z, symbol):
        self.target[x][y][z] = symbol


class RawTransfer:
    def __init__(self) -> None:
        self.channel = Channel()
        self.r = 3  # repeat the symbol 3 times

    def transfer(self):
        for (x, y, z, symbol) in self.channel.send():
            self.channel.decode(x, y, z, self.channel.transfer(symbol))
        Image.fromarray(self.channel.target).save('cover_with_noise.png')


class RepetitionCodes:
    def __init__(self) -> None:
        self.channel = Channel()
        self.r = 3  # repeat the symbol 3 times

    def transfer(self):
        for (x, y, z, symbol) in self.channel.send():
            symbols = defaultdict(int)
            for _ in range(self.r):
                symbols[self.channel.transfer(symbol)] += 1
            score = max(symbols.items(), key=lambda x: x[1])
            symbol = score[0]
            self.channel.decode(x, y, z, symbol)
        Image.fromarray(self.channel.target).save('repetition_codes.png')

class HammingCode:
    def __init__(self) -> None:
        self.G = np.asarray([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
        ])
        self.H = np.asarray([
            [1, 1, 1, 0, 1, 0, 0,],  # P I_3
            [0, 1, 1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 1]
        ])
        self.lookup = {
            "000": None,
            "001": 6,
            "010": 5,
            "011": 3,
            "100": 4,
            "101": 0,
            "110": 1,
            "111": 2,
        }
        self.channel = Channel()

    def create_t_vector(self, vector):
        #print(vector.shape)
        #print(self.G.shape)
        return (self.G @ vector) % 2
    
    def recover_transferred_vector(self, transformed):
        decode = "".join(map(str, ((self.H @ transformed) % 2).reshape(-1).tolist()))
        decode_index = self.lookup[decode]
        if decode_index is not None:
            transformed[decode_index] = (transformed[decode_index] + 1) % 2
        decode = transformed[:4]
        return decode
    
    def test_flip(self, vector, flip):
        assert vector.shape[-1] == 4
        transformed = self.create_t_vector(vector)
        # flip the bit
        if flip != None:
            transformed[flip] = (transformed[flip] + 1) % 2
        decode = self.recover_transferred_vector(transformed)
        assert np.all(decode == vector) 

    def transfer_symbol(self, symbol):
        assert symbol >= 0 and symbol <= 255
        def bitfield(n):
            size = bin(n)[2:]
            size = '0' * (8 - len(size)) + size
            output = np.asarray([int(digit) for digit in size])
            assert output.shape[0] == 8, output.shape[0]
            return output
        bit_vector = bitfield(symbol)
        first_half = self.create_t_vector(bit_vector[:4])
        second_half = self.create_t_vector(bit_vector[4:])
        return (first_half, second_half)

    def apply_noise(self, first_half, second_half):
        def transfer(n):
            for i in range(n.shape[0]):
                if np.random.rand() <= self.channel.p_flip:
                    n[i] = (n[i] + 1) % 2
            return n
        # transfer
        first_half = transfer(first_half)
        second_half = transfer(second_half)
        return first_half,  second_half
    
    def decode_vector(self, first_half, second_half):
        first_half = self.recover_transferred_vector(first_half).tolist()
        second_half = self.recover_transferred_vector(second_half).tolist()
        bits_to_int = int("".join(map(str, first_half + second_half)), 2)
        return bits_to_int
    
    def test_encode_decode(self, symbol):
        first_half, second_half = self.transfer_symbol(symbol)
        first_half, second_half = self.apply_noise(first_half, second_half)        
        bits_to_int = self.decode_vector(first_half, second_half)
        return bits_to_int

    def transfer(self):
        for (x, y, z, symbol) in self.channel.send():
            bits_to_int = self.test_encode_decode(symbol)
           # assert symbol == bits_to_int, f"{symbol} {bits_to_int}"
            print((symbol, bits_to_int))

            self.channel.decode(x, y, z, bits_to_int)
        Image.fromarray(self.channel.target).save('hamming_codes.png')

    def transfer_bits(self, vector):
        pass

if __name__ == "__main__":
    #    a = RawTransfer()
    #    a.transfer()

    #    a = RepetitionCodes()
    #    a.transfer()

    a = HammingCode()
    a.test_flip(np.asarray([1, 0, 0,0]), 1)
    a.test_flip(np.asarray([0, 0, 1,0]), 1)
    a.test_flip(np.asarray([0, 1, 1,0]), 1)
    a.test_flip(np.asarray([1, 1, 0,1]), 1)
    a.test_flip(np.asarray([0, 1, 1,0]), None)
    assert a.test_encode_decode(128) == 128
    assert a.test_encode_decode(200) == 200
    assert a.test_encode_decode(64) == 64
    assert a.test_encode_decode(105) == 105
    a.transfer()
