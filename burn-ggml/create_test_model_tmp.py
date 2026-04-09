
import struct

def create_gguf_model(path):
    # GGUF Magic
    magic = b"GGUF"
    version = 3
    n_tensors = 1
    n_kv = 1
    
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))
        
        # KV: general.alignment
        key = b"general.alignment"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", 4)) # Type: Uint32
        f.write(struct.pack("<I", 32))
        
        # Tensor Info
        name = b"test.weight"
        f.write(struct.pack("<Q", len(name)))
        f.write(name)
        f.write(struct.pack("<I", 2)) # 2 dims
        f.write(struct.pack("<Q", 2)) # 2
        f.write(struct.pack("<Q", 2)) # 2
        f.write(struct.pack("<I", 0)) # Type: F32
        f.write(struct.pack("<Q", 0)) # offset
        
        # Alignment padding
        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        f.write(b"\0" * pad)
        
        # Tensor Data
        for val in [1.0, 2.0, 3.0, 4.0]:
            f.write(struct.pack("<f", val))

if __name__ == "__main__":
    create_gguf_model("test_model.gguf")
