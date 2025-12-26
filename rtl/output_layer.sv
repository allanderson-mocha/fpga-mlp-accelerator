import nn_arch_pkg::*;
`timescale 1ns/1ps

module output_layer (
    input logic                         clk,
    input logic                         rst_n,
    input logic [ACC_W*HIDDEN_SIZE-1:0] hidden_in_flat,
    input logic                         start,

    input data_t weight[HIDDEN_SIZE][OUTPUT_SIZE],
    input acc_t  bias[OUTPUT_SIZE],

    output logic [3:0]             class_idx,
    output logic [OUTPUT_SIZE-1:0] one_out,
    output logic                   done
);

    logic finished;
    logic start_d;
    logic start_pulse;
    acc_t logits[OUTPUT_SIZE];
    acc_t hidden_in[HIDDDEN_SIZE];
    
    // Ensure start signal is only a pulse
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) start_d <= 1'b0;
        else        start_d <= start;

    assign start_pulse = start & ~start_d;

    // Unpack hidden output signal
    genvar i;
    generate
        for (i = 0; i < HIDDEN_SIZE; i++) begin : UNPACK_HIDDEN
            assign hidden_in[i] =
                hidden_in_flat[(i+1)*ACC_W-1 -: ACC_W];
        end
    endgenerate


    // ---------------- Output Core ----------------
    output_core u_oc (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_pulse),
        .h_in(hidden_in),
        .weight(weight),
        .bias(bias),
        .logits(logits),
        .finished(finished)
    );

    // Argmax operation
    logic [3:0] argmax_idx;
    logic [OUTPUT_SIZE-1:0] argmax_onehot;

    argmax_comb u_argmax (
        .logits(logits),
        .class_idx(argmax_idx),
        .onehot(argmax_onehot)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            class_idx <= '0;
            one_out   <= '0;
        end
        else if (finished) begin
            class_idx <= argmax_idx;
            one_out   <= argmax_onehot;
        end
    end

    assign done = finished;
endmodule