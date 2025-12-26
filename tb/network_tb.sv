import nn_arch_pkg::*;
`timescale 1ns/1ps

module tb_phase1;
    parameter CLK_PERIOD = 10;

    // DUT signals
    logic clk;
    logic rst_n;
    logic start;
    logic [DATA_W*IN_DIM-1:0] bus_in;
    
    data_t weight_h[HIDDEN_SIZE][IN_DIM];
    data_t weight_o[HIDDEN_SIZE][OUTPUT_SIZE];
    acc_t  bias_h[HIDDEN_SIZE];
    acc_t  bias_o[OUTPUT_SIZE];

    logic [3:0] class_idx;
    logic [OUTPUT_SIZE-1:0] one_out;
    logic hidden_all_done;
    logic output_done;
    logic [ACC_W*HIDDEN_SIZE-1:0] hidden_out_flat;

    // Unpacked array for easier access
    acc_t hidden_out[0:7];

    // Clock generation
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Instantiate DUT
    network_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .bus_in(bus_in),
        .weight_h(weight_h),
        .weight_o(weight_o),
        .bias_h(bias_h),
        .bias_o(bias_o),
        .class_idx(class_idx),
        .one_out(one_out),
        .hidden_all_done(hidden_all_done),
        .hidden_out_flat(hidden_out_flat),
        .output_done(output_done)
    );

    // Unpack flat wires
    genvar h;
    generate
        for (h = 0; h < 8; h = h + 1)
            assign hidden_out[h] = hidden_out_flat[(h+1)*ACC_W-1 -: ACC_W];
    endgenerate

    // Test vectors
    logic [DATA_W*IN_DIM-1:0] test_vectors [0:3];

    // Expected values
    logic signed [ACC_W-1:0] expected_hidden [0:7];
    acc_t expected_logits [OUTPUT_SIZE];
    logic [OUTPUT_SIZE-1:0] exp_onehot;

    // Task to compute expected outputs
    task automatic compute_expected;
        input [DATA_W*IN_DIM-1:0] in_vector;
        integer i, j;
        int max_idx;
        acc_t max_val;
        acc_t acc;
        data_t w, x;
        begin
            // Hidden layer
            for (i = 0; i < 8; i = i + 1) begin
                acc = bias_h[i];
                for (j = 0; j < IN_DIM; j = j + 1) begin
                    w = weight_h[i][j];
                    x = in_vector[(j+1)*DATA_W-1 -: DATA_W];
                    acc = acc + $signed({{ACC_W-DATA_W{w[DATA_W-1]}}, w}) *
                                $signed({{ACC_W-DATA_W{x[DATA_W-1]}}, x});
                end
                expected_hidden[i] = (acc < 0) ? 0 : acc; // ReLU
            end

            // Output layer
            for (i = 0; i < 10; i = i + 1) begin
                acc = bias_o[i];
                for (j = 0; j < 8; j = j + 1) begin
                    w = weight_o[j][i];
                    acc = acc + $signed({{(ACC_W-DATA_W){w[DATA_W-1]}}, w}) * $signed(expected_hidden[j]);
                end
                expected_logits[i] = acc;
            end
            
            // One hot expectation
            max_idx = 0;
            max_val = expected_logits[0];

            for (int i = 1; i < OUTPUT_SIZE; i++) begin
                if (expected_logits[i] > max_val) begin
                    max_val = expected_logits[i];
                    max_idx = i;
                end
            end

            exp_onehot = '0;
            exp_onehot[max_idx] = 1'b1;
        end
    endtask

    // Initialize test vectors
    initial begin
        test_vectors[0] = {64{8'd0}}; // all zeros
		test_vectors[1] = {64{8'd1}}; // all ones
		test_vectors[2] = {
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8,
			8'd1,8'd2,8'd3,8'd4,8'd5,8'd6,8'd7,8'd8
		};
		test_vectors[3] = {
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1,
			8'd8,8'd7,8'd6,8'd5,8'd4,8'd3,8'd2,8'd1
		};

        // Reset
        rst_n = 0; start = 0; bus_in = 0;

        // Load weights
        $readmemh("W1_q.mem", weight_h);
        $readmemh("W2_q.mem", weight_o);
        $readmemh("b1_q.mem", bias_h);
        $readmemh("b2_q.mem", bias_o);

        #(CLK_PERIOD*2);
        rst_n = 1;

        // Run tests
        for (integer t = 0; t < 4; t = t + 1) begin
            bus_in = test_vectors[t];
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;

            wait(output_done);
			@(posedge clk);

            compute_expected(test_vectors[t]);

            if (one_out !== exp_onehot) begin
                $error("FAIL t=%0d expected=%b got=%b (class_idx=%0d)", t, exp_onehot, one_out, class_idx);
                end else begin
                $display("PASS t=%0d class=%0d onehot=%b", t, class_idx, one_out);
            end

            // Display
            $display("\n=== Test %0d ===", t+1);
            $write("Hidden outputs (DUT): ");
            for (integer j = 0; j < 8; j = j + 1) $write("%0d ", hidden_out[j]);
            $write("\n");
            if (t == 0) begin
                $write("Hidden expected     : ");
                for (integer j = 0; j < 8; j = j + 1) $write("%0d ", expected_hidden[j]);
                $write("\n");
            end

            $display("Expexted one-hot = %b", exp_onehot);            
            $display("Predicted class = %0d", class_idx);
            $display("One-hot output  = %b", one_out);

            #(CLK_PERIOD*5);
        end

        $display("\nAll tests finished.");
    end

    // Dump waves
    initial begin
        $dumpfile("tb_phase1.vcd");
        $dumpvars(0, tb_phase1);
    end

endmodule
