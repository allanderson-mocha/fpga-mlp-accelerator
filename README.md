# Neural Network RTL Accelerator (Single Hidden Layer)

## Overview

This repository implements a **parameterized, synthesizable neural network accelerator** in SystemVerilog.  
The design targets **inference only** and consists of:

- One **fully parallel hidden layer**
- One **time-multiplexed output (fully connected) layer**
- A clean **valid-based control protocol**
- A **self-checking testbench** with a golden reference model

---

## Network Architecture

```bash
Input Vector (packed bus)
|
v
+------------------+
| Hidden Layer | (parallel hidden nodes)
| - MAC + bias |
| - ReLU |
+------------------+
|
| hidden_out (packed)
v
+------------------+
| Output Layer | (time-multiplexed MAC core)
| - MAC + bias |
| - Argmax |
+------------------+
|
v
Class Index + One-Hot Output
```

### Layer Summary

| Layer  | Implementation Style           | Parallelism                                |
| ------ | ------------------------------ | ------------------------------------------ |
| Hidden | `hidden_layer`                 | Fully parallel (`HIDDEN_SIZE` nodes)       |
| Output | `output_layer` + `output_core` | Time-multiplexed (2 outputs per iteration) |

---

## Data Types and Parameters

All global parameters and type definitions live in `nn_arch_pkg`.

Parameters include:

- `IN_DIM` – Input vector length
- `HIDDEN_SIZE` – Number of hidden neurons
- `OUTPUT_SIZE` – Number of output classes
- `DATA_W` – Input/weight bit width
- `ACC_W` – Accumulator bit width

Type aliases:

- `data_t` – Signed input/weight type (e.g. 8-bit)
- `acc_t` – Signed accumulator type (e.g. 64-bit)

---

## Module Breakdown

### `network_top`

Top-level integration module.

**Responsibilities**

- Connects hidden and output layers
- Routes weights and biases
- Generates control pulses between layers
- Exposes final classification outputs

**Key Signals**

- `start` – One-cycle pulse to begin inference
- `hidden_all_done` – Indicates hidden layer completion
- `output_done` – Indicates classification is valid
- `class_idx` – Predicted class index
- `one_out` – One-hot encoded prediction

---

### `hidden_layer`

Implements the hidden layer.

**Characteristics**

- Fully parallel: one `hidden_node` per hidden neuron
- Accepts packed input bus, unpacks internally
- Applies MAC + bias + ReLU
- Outputs remain stable after completion

**Contract**

- `hidden_out` is valid and stable when `hidden_all_done == 1`
- Values remain stable until the next `start`

---

### `hidden_node`

Single hidden neuron compute unit.

**Behavior**

- Sequential MAC over input dimension
- Adds bias
- Applies ReLU
- Asserts `done` when output is valid

---

### `output_layer`

Completed output layer wrapper.

**Responsibilities**

- Accepts packed hidden outputs
- Unpacks and feeds compute core
- Edge-detects `start`
- Registers argmax results on completion

This module defines a **clean architectural boundary** between computation and decision logic.

---

### `output_core`

Low-level compute engine for the output layer.

**Characteristics**

- Time-multiplexed MAC FSM
- Computes two output logits per iteration
- Uses unpacked arrays for clarity
- Produces a one-cycle `finished` pulse

This separation allows:

- Easy reuse
- Future pipelining
- Clear timing analysis

---

### `argmax_comb`

Purely combinational argmax.

**Notes**

- Operates on stable logits
- Outputs are registered by `output_layer`
- Avoids glitching and combinational hazards

---

## Control and Timing Model

The design uses a **valid-based, pulse-driven control scheme**.

### Inference Sequence

1. `start` asserted for **one cycle**
2. Hidden nodes compute in parallel
3. `hidden_all_done` asserts
4. Output layer begins computation
5. `output_done` asserts for one cycle
6. Outputs are registered and stable

### Important Guarantees

- All data is **observed only when valid**
- Outputs remain stable after `done`
- No combinational control feedback paths

---

## Testbench (`tb_phase1`)

The testbench is **self-checking** and follows industry verification practices.

### Key Features

- Drives packed input vectors
- Loads weights and biases from `.mem` files
- Waits on `output_done` before sampling
- Computes **expected one-hot output**
- Compares DUT vs expected results
- Generates waveforms (`.vcd`)

### Golden Model Strategy

- Hidden layer and output layer are recomputed in TB
- Argmax is computed in TB
- Only **final classification** is compared

This avoids false failures due to:

- Intermediate quantization
- Internal pipeline timing
- Microarchitectural differences

---

## Memory Handling

Weights and biases are:

- Loaded in the **testbench**
- Passed into the DUT via ports

This cleanly separates:

- **Design logic** (RTL)
- **Configuration data** (TB / environment)

---

## Synthesis Notes

- Core RTL is fully synthesizable
- `$readmemh` is used **only in the testbench**
- Unpacked arrays are used internally for clarity
- Packed buses are used at module boundaries

---

## Design Philosophy

This project emphasizes:

- Clear separation of concerns
- Explicit control/data contracts
- Valid-based synchronization
- Readable, reviewable RTL
- Scalable parameterization

---

## Possible Extensions

- Pipelined output core
- Valid/ready handshake
- Multi-layer support
- Streaming input interface
- Backpressure support
- SRAM-based weight storage

---

## Summary

This design implements a clean, modular neural network accelerator with:

- Well-defined timing and control
- Industry-standard RTL patterns
- Robust verification methodology

It is suitable as both a **working inference engine** and a **reference-quality RTL example**.
