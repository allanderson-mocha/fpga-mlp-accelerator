import nn_aarch_pkg::*;
`timescale 1ns/1ps

module network_top (
    input logic                     clk,
    input logic                     rst_n,
    input logic [DATA_W*IN_DIM-1:0] bus_in,
    input logic                     start,
    input data_t                    weight_h[HIDDEN_SIZE][IN_DIM],
    input data_t                    weight_o[HIDDEN_SIZE][OUTPUT_SIZE],
    input acc_t                     bias_h[HIDDDEN_SIZE],
    input acc_t                     bias_o[OUTPUT_SIZE],

    output logic [3:0]                   class_idx,
    output logic [OUTPUT_SIZE-1:0]       one_out,
    output logic                         hidden_all_done,
    output logic [ACC_W*HIDDEN_SIZE-1:0] hidden_out_flat,
    output logic                         output_done
);

    // Hidden Layer
    hidden_layer u_hidden (
        .clk(clk),
        .rst_n(rst_n),
        .bus_in(bus_in),
        .start(start),
        .weight(weight_h),
        .bias(bias_h),
        .hidden_out_flat(hidden_out_flat),
        .hidden_all_done(hidden_all_done)
    )

    // Output Layer
    // Hidden done stability protection
    logic hidden_all_done_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) hidden_all_done_d <= 1'b0;
        else hidden_all_done_d <= hidden_all_done;
    end

    logic output_layer_start 
    assign output_layer_start = hidden_all_done & ~hidden_all_done_d;

    output_layer u_fc (
        .clk(clk),
        .rst_n(rst_n),
        .start(output_layer_start),
        .hidden_in_flat(hidden_out),
        .weight(weight_o),
        .bias(bias_o),
        .class_idx(class_idx),
        .one_out(one_out),
        .done(output_done)
    );

endmodule

