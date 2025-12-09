
# pack_W1_for_verilog.py
# Input: W1_q.mem with 512 lines (64 weights per hidden node, 8 hidden nodes)
# Output: W1_q_packed.mem with 8 lines, each line is 64 bytes (for Verilog hidden_weight[0:7])

input_file = "W1_q.mem"
output_file = "W1_q_packed.mem"

# Read all lines and clean
with open(input_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# Check total number of lines
if len(lines) != 64 * 8:
    raise ValueError(f"Expected 512 lines (64x8), got {len(lines)}")

# Pack per hidden node
packed_lines = []
for node in range(8):
    node_weights = lines[node*64:(node+1)*64]
    # Concatenate into one line for Verilog (MSB first)
    packed_line = ''.join(node_weights)
    packed_lines.append(packed_line)

# Write to new file
with open(output_file, "w") as f:
    for line in packed_lines:
        f.write(line + "\n")

print(f"Done! Packed {len(packed_lines)} lines into {output_file}")
