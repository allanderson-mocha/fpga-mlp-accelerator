# compute_golden_logits.py
import numpy as np
import sys
from pathlib import Path

# -------------------------
# Utility to load signed mem files robustly
# -------------------------
def load_signed_mem(path, bit_width):
    vals = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # Handle negative hex manually
            if s.startswith('-'):
                val = -int(s[1:], 16)
            else:
                val = int(s, 16)

            # Convert to signed range e.g. int8 → -128..127
            max_val = 2**(bit_width-1)
            val = (val + max_val) % (2*max_val) - max_val  
            vals.append(val)

    return np.array(vals)


# -------------------------
# Main: shape constants (your design)
# -------------------------
IN_DIM = 64
HIDDEN = 8
OUT = 10
DATA_W = 8  # bit-width of hidden outputs and weight elements
ACC_W_BIAS = 40  # bias width in HDL (we'll keep biases in 64-bit python container)

# -------------------------
# Load mem files (assumes these filenames)
# -------------------------
base = Path('.')  # adjust if mems are elsewhere
W1_path = base / "W1_q.mem"   # 64 x 8, int8
b1_path = base / "b1_q.mem"   # 8 entries, int32 (in mem saved as 8 hex digits)
W2_path = base / "W2_q.mem"   # 8 x 10, int8
b2_path = base / "b2_q.mem"   # 10 entries, int32

if not W1_path.exists() or not b1_path.exists() or not W2_path.exists() or not b2_path.exists():
    print("ERROR: could not find one of the mem files in current directory.")
    print("Expected: W1_q.mem, b1_q.mem, W2_q.mem, b2_q.mem")
    sys.exit(1)

W1_flat = load_signed_mem(W1_path, 8)   # int8 values
b1_flat = load_signed_mem(b1_path, 32)  # int32 values
W2_flat = load_signed_mem(W2_path, 8)
b2_flat = load_signed_mem(b2_path, 32)

assert W1_flat.size == IN_DIM * HIDDEN, f"W1 size mismatch: {W1_flat.size}"
assert b1_flat.size == HIDDEN, f"b1 size mismatch: {b1_flat.size}"
assert W2_flat.size == HIDDEN * OUT, f"W2 size mismatch: {W2_flat.size}"
assert b2_flat.size == OUT, f"b2 size mismatch: {b2_flat.size}"

W1 = W1_flat.reshape(IN_DIM, HIDDEN)   # shape (64,8) -> matches original Python training shape
b1 = b1_flat.reshape(HIDDEN)           # shape (8,)
W2 = W2_flat.reshape(HIDDEN, OUT)      # shape (8,10)
b2 = b2_flat.reshape(OUT)              # shape (10,)

# -------------------------
# Helper to convert a bus_in (64 bytes) -> numpy vector used by RTL
# We mimic RTL pipeline:
#   - bus_in is treated as signed 8-bit integers (the HDL uses 8-bit multiplies)
#   - hidden accumulator: Z1_int32 = X_q @ W1 + b1  (int32)
#   - ReLU applied on Z1_int32 (negative -> 0)
#   - hidden_out for RTL is lower DATA_W bits of Z1_int32 (i.e., Z1_int32 & 0xff) after ReLU
#   - output logits (int32) = hidden_out_hw @ W2 + b2
# -------------------------
def rtl_like_inference(bus_in_bytes):
    # bus_in_bytes: list/np array length IN_DIM with values 0..255 or -128..127
    Xq = np.array(bus_in_bytes, dtype=np.int64)

    # Convert Xq to signed 8-bit range if >127 (unsigned read from hex)
    # assume elements are in 0..255; map >127 to negative signed
    Xq_signed = Xq.copy()
    Xq_signed[Xq_signed >= 128] -= 256
    # compute hidden pre-accum (int64 for safety)
    # Z1_int32 = X_q (1x64) @ W1 (64x8) + b1  -> shape (8,)
    Z1 = (Xq_signed.astype(np.int64) @ W1.astype(np.int64)) + b1.astype(np.int64)
    # ReLU
    A1 = np.maximum(Z1, 0)

    # Now mimic RTL: hidden output is lowest DATA_W bits of accumulator after ReLU
    # i.e. mimic: out <= acc[DATA_W-1:0] from HDL (acc is integer)
    # We take A1 modulo 2**DATA_W, then interpret as signed 8-bit (two's complement)
    mask = (1 << DATA_W) - 1
    hidden_out_hw = (A1.astype(np.int64) & mask).astype(np.int64)
    # interpret as signed 8-bit:
    hidden_out_hw[hidden_out_hw >= (1 << (DATA_W-1))] -= (1 << DATA_W)

    # Now compute output layer logits: (1x8) @ (8x10) + b2  -> shape (10,)
    Z2 = (hidden_out_hw.astype(np.int64) @ W2.astype(np.int64)) + b2.astype(np.int64)

    return {
        "Xq_signed": Xq_signed,
        "Z1_int32": Z1.astype(np.int64),
        "A1_int32": A1.astype(np.int64),
        "hidden_out_hw": hidden_out_hw.astype(np.int64),
        "Z2_int32": Z2.astype(np.int64)
    }

# -------------------------
# TWO USAGE EXAMPLES
# 1) Dummy deterministic vector: all 127 (so quantized input = 127)
# 2) Use the testbench bus_in (copy/paste bytes) – user can replace
# -------------------------
# Example A: dummy test (recommended for golden test)
dummy_bus = np.array([127] * IN_DIM, dtype=np.int64)  # matches Python quantization when Xf32 = ones
res = rtl_like_inference(dummy_bus)

print("=== GOLDEN (DUMMY) TEST: bus_in = all 127 ===")
print("Hidden pre-accum (Z1_int32):")
print(res["Z1_int32"].tolist())
print("\nHidden post-ReLU (A1_int32):")
print(res["A1_int32"].tolist())
print("\nHidden outputs seen by RTL (signed 8-bit):")
print(res["hidden_out_hw"].tolist())
print("\nFinal integer logits (Z2_int32) -- paste these into your testbench as expected values:")
print(res["Z2_int32"].tolist())

# Example B: if you want to verify your current random TB inputs, you can fill bus_in_tb below
# and re-run the script. Example format for bus_in_tb: [0x10, 0x22, ...] (64 entries)
# bus_in_tb = [0x10, 0x22, ...]  # replace with your TB values
# res_tb = rtl_like_inference(bus_in_tb)
# print(res_tb["Z2_int32"].tolist())
