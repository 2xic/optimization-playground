"""
https://wwwconference.org/wp-content/uploads/2025/01/paper215.pdf
https://www.fromkk.com/posts/near-duplicate-with-simhash/
"""
from collections import defaultdict
import hashlib

document_a = """
hello world, how are you doing ? 
"""
document_b = """
hello world, 


how are you doing!!


Yeah, this is some new text also.
"""

def _string_hash(token, hash_bits):
    token_hash = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
    return token_hash & ((1 << hash_bits) - 1)

def get_fingerprint(doc: str, window_size=4):
    sub_string_counts = defaultdict(int)
    # pre processing
    doc = doc.replace(" ", "")
    for i in range(0, len(doc)):
        sub_string_counts[doc[i:i+window_size]] += 1

    vector = [0] * window_size

    for token, weight in sub_string_counts.items():
        # Hash the token to get a binary representation
        token_hash = _string_hash(token, window_size)
        
        # Update the vector using the token hash
        for i in range(window_size):
            # Check if the i-th bit in the token hash is set
            bit = (token_hash >> i) & 1
            # Update the vector: add `weight` if the bit is 1, subtract if 0
            vector[i] += weight if bit else -weight
    
    # Generate the SimHash fingerprint by taking the sign of each vector component
    fingerprint = 0
    for i in range(window_size):
        if vector[i] > 0:
            fingerprint |= (1 << i)  # Set the i-th bit if the value is positive

    return fingerprint

def delta(hash1, hash2):
    x = hash1 ^ hash2 
    return bin(x).count('1')

if __name__ == "__main__":
    print(get_fingerprint(document_a))
    print(get_fingerprint(document_b))
    print(delta(
        get_fingerprint(document_a),
        get_fingerprint(document_b),
    ))
