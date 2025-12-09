`timescale 1ns/1ps

module phase1_tb;
parameter IN_DIM = 64;
parameter DATA_W = 8;

reg clk;
reg rst_n;
reg start;
reg [DATA_W*IN_DIM-1:0] bus_in;

wire [3:0] class_idx;
wire [9:0] one_out;
wire [7:0] hidden_done;
wire [7:0][DATA_W-1:0] hidden_out;
wire hidden_all_done;
wire hidden_all_done_d;
wire output_layer_start;
wire signed [31:0] logits[9:0];

// Instantiate DUT
phase1 dut(
    .clk(clk),
    .rst_n(rst_n),
    .bus_in(bus_in),
    .start(start),
    .class_idx(class_idx),
    .one_out(one_out)
);

// Clock generation
initial clk = 0;
always #5 clk = ~clk;

// Reset
initial begin
    rst_n = 0;
    start = 0;
    bus_in = 0;
    #20 rst_n = 1;
end

// Task to apply test vector and wait for output layer to finish
task apply_vector(input [DATA_W*IN_DIM-1:0] vec);
    begin
        bus_in = vec;
        @(posedge clk); start = 1;
        @(posedge clk); start = 0;  // single-cycle pulse

        // Wait for hidden layer to finish
        wait(dut.hidden_all_done);
        // Wait for output layer FSM to finish
        wait(dut.output_finished);

        #5;  // small delay to allow outputs to settle
        $display("Time %0t | Hidden Out: %p", $time, hidden_out);
        $display("Time %0t | Logits: %p", $time, logits);
        $display("Time %0t | Class Index: %d | One-hot: %b", $time, class_idx, one_out);
    end
endtask

// Test vectors (same as original Verilog TB)
reg [DATA_W*IN_DIM-1:0] test_vector1;
reg [DATA_W*IN_DIM-1:0] test_vector2;

initial begin
    // Fill test_vector1 (Random Input)
    test_vector1 = {
        8'h10,8'h22,8'hF3,8'h01,8'hB0,8'h7F,8'h22,8'h55,
        8'h11,8'h49,8'h66,8'h24,8'h80,8'h90,8'hAA,8'h12,
        8'h07,8'hCD,8'hFE,8'h0F,8'h89,8'hAB,8'h04,8'h33,
        8'h05,8'h08,8'h99,8'h34,8'h77,8'h55,8'h22,8'hF2,
        8'h01,8'h33,8'h55,8'h77,8'h99,8'hBB,8'hCC,8'hDD,
        8'h5A,8'h6B,8'h7C,8'h8D,8'h9E,8'hAF,8'hBE,8'hCF,
        8'hFE,8'h11,8'h22,8'h09,8'h10,8'h20,8'h40,8'h09,
        8'h88,8'h77,8'h66,8'h55,8'h44,8'h33,8'h22,8'h11
    };

    // Fill test_vector2 (All 0x7F)
    test_vector2 = {64{8'h7F}};

    #30;
    $display("=== TEST 1: RANDOM INPUT ===");
    apply_vector(test_vector1);

    #30;
    $display("=== TEST 2: ALL 0x7F INPUT ===");
    apply_vector(test_vector2);

    #50 $finish;
end

endmodule