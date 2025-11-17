module phase1 #(
    parameter IN_DIM = 64, parameter DATA_W = 8
)(
    input   wire                      clk,
    input   wire                      rst_n,    // Active-low reset
    input   wire [DATA_W*IN_DIM-1:0]  bus_in,
    output  wire [9:0]                out,
);
    
endmodule

// Maybe create module for hidden layer node
module hidden_node #(
    parameter IN_DIM = 64, 
    parameter DATA_W = 8, 
    parameter ACC_W = 40
) (
    input   wire                      clk,
    input   wire                      rst_n,    // Active-low reset
    input   wire                      start,    // One cycle pulse
    input   wire [DATA_W*IN_DIM-1:0]  in,
    input   wire [ACC_W-1:0]          bias,
    input   wire [DATA_W*IN_DIM-1:0]  weight,
    
    output  reg [DATA_W-1:0]          out,
    output  reg                       done
);

// FSM state definition
typedef enum logic [2:0] {
        IDLE = 3'd0,
        MULT = 3'd1,
        SUM = 3'd2,
        DONE = 3'd3
} state_t;

state_t state, next_state;

// Internal registers
reg [$clog2(IN_DIM)-1:0] idx;
reg [2*DATA_W-1:0]       product;
reg [ACC_W-1:0]          acc;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        acc <= 0;
        product <= 0;
        idx <= 0;
        done <= 1'b0;
    end else begin
        state <= next_state;

        case(state)
            IDLE: begin
                done <= 1'b0;
                // Wait for pulse
                if (start) begin
                    idx <= 0;
                    acc <= bias;
                end
            end
            
            MULT: begin
                product <= in[idx*DATA_W +: DATA_W] * weight[idx*DATA_W +: DATA_W];
            end

            SUM: begin
                acc <= acc + product;
                idx <= idx + 1;
            end

            DONE: begin
                done <= 1'b1;

                //ReLU
                if(acc[ACC_W-1] == 1'b1) begin
                    out <= 0;
                end else begin
                    out <= acc[DATA_W-1:0];
                end
            end

            default: ;
        endcase
    end
end

always @(*) begin
    next_state = state;

    case (state)
        IDLE: begin
            if (start)
                next_state = MULT;
        end

        MULT: begin
            next_state = SUM;
        end

        SUM: begin
            if (idx == IN_DIM-1)
                next_state = DONE;
            else
                next_state = MULT;
        end

        DONE: begin
            if (!start)
                next_state = IDLE;
        end

        default: next_state = IDLE;
    endcase
end
endmodule