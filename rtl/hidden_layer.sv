import nn_arch_pkg::*;
`timescale 1ns/1ps

module hidden_layer (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic [DATA_W*IN_DIM-1:0] bus_in, // Packed bus input
    input  logic                     start,

    input  data_t weight[HIDDEN_SIZE][IN_DIM],
    input  acc_t  bias[HIDDEN_SIZE],

    output logic [ACC_W*HIDDEN_SIZE-1:0] hidden_out_flat,
    output logic                         hidden_all_done
);

    logic [HIDDEN_SIZE-1:0] hidden_done;
    acc_t  hidden_out[HIDDEN_SIZE];
    data_t in[IN_DIM];

    // Start pulse protection
    logic start_d, start_pulse;

    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) start_d <= 1'b0;
        else        start_d <= start;

    assign start_pulse = start & ~start_d;


    // Unpack input bus for nodes
    genvar j;
    generate
        for (j = 0; j < IN_DIM; j++) begin : UNPACK_INPUT
            assign in[j] =
                bus_in[(j+1)*DATA_W-1 -: DATA_W];
        end
    endgenerate

    // Instantiate all hidden nodes
    genvar i;
    generate
        for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin : HIDDEN_LAYER
            hidden_node u_hidden (
                .clk(clk),
                .rst_n(rst_n),
                .start(start_pulse),
                .in(in),
                .weight(weight[i]),
                .bias(bias[i]),
                .out(hidden_out[i]),
                .done(hidden_done[i])
            );
        end
    endgenerate

    assign hidden_all_done = &hidden_done;

    // Flatten output
    generate
        for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin : HIDDEN_DBG
            assign hidden_out_flat[(i+1)*ACC_W-1 -: ACC_W] = hidden_out[i];
        end
    endgenerate
endmodule
