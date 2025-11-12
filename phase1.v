module phase1 #(parameter IN_DIM = 64, parameter DATA_W = 8)(
    input   reg [DATA_W*IN_DIM-1:0]  bus_in;
    output  reg [9:0]                    out;
);
    
endmodule

// Maybe create module for hidden layer node
module hidden_node #(parameter IN_DIM = 64, parameter DATA_W = 8, parameter ACC_W = 40) (
    input   reg [DATA_W*IN_DIM-1:0]    in;
    input   reg [ACC_W-1:0]            bias;
    input   reg [DATA_W-1:0]           weight;
    output  reg [DATA_W-1:0]           out;
);

    

endmodule