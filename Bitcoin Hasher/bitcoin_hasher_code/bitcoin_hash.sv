module bitcoin_hash (input logic        clk, reset_n, start,
                     input logic [15:0] message_addr, output_addr,
                    output logic        done, mem_clk, mem_we,
                    output logic [15:0] mem_addr,
                    output logic [31:0] mem_write_data,
                     input logic [31:0] mem_read_data);
parameter num_nonces = 16;
//This design offers two modes: one mode with less parellelism (instantiations of half the amount of nonces), and another with instantions equal to the full amount of nonces
//BY DEFAULT: we are at the mode where the number of sha instantions is equal to the number of nonces, so all the hashes for all the nonces are done AT ONCE for each of stages 2 and 3.
//So more parallelism and less cycles vs. less parallelism but more cycles 
// These parameters determine multiple things: mode 0 is the mode with instantiations of the sha module equal to the amount of nonces,
//so, each stage of the bitcoin hashing is done once not twice. At mode 1, the instantiations are half the amount of nonces and each stage is done twice
//once for nonces 0-7 then another stage for nonces 8-15. At mode 0, stages 21 and 31 aren't used, but at mode 1 they are used
// the rest of the parameters following mode are used as constants in for loops and indexes of arrays used in this design
// They are further clarified in the report.
parameter mode = 0;
parameter para1 = (mode==1) ? num_nonces/2:num_nonces;
parameter hash_complete_para = (mode==1) ? num_nonces/2:0;

logic begin_stage1;
logic end_stage1;
logic begin_stage2;
logic end_stage2[para1];
logic [15:0] cur_addr;
logic [15:0] offset;
logic        cur_we;
logic [31:0] cur_write_data;
logic [31:0] w_block[para1][16]; //words to be used in sha instantions
logic [31:0] message[19];
logic [31:0] hash_complete[num_nonces][8]; //will contain the hashes of all nonces once bitcoin hasher completes all stages
logic [31:0] hash_in[8];// hashes used in the instantiations 
logic [31:0] hash_out[para1][8]; //hashes out of the instantiations
logic [31:0] hash1[8];  
logic [7:0] i;

//The states of this FSM are 8. IDLE, READ, and WRITE are familiar ones.
//Stage11 is the first stage of bitcoin hashing requiring one instantiation of sha. Stage 21 and 22, and 31 and 32 are the two phases of stages 2 and 3
//when only HALF the number of nonces are instantiated which occurs at mode 1
enum logic [3:0] {IDLE, READ, STAGE11, STAGE21, STAGE22, STAGE31, STAGE32, WRITE} state, next_state;

parameter int k[64] = '{
    32'h428a2f98,32'h71374491,32'hb5c0fbcf,32'he9b5dba5,32'h3956c25b,32'h59f111f1,32'h923f82a4,32'hab1c5ed5,
    32'hd807aa98,32'h12835b01,32'h243185be,32'h550c7dc3,32'h72be5d74,32'h80deb1fe,32'h9bdc06a7,32'hc19bf174,
    32'he49b69c1,32'hefbe4786,32'h0fc19dc6,32'h240ca1cc,32'h2de92c6f,32'h4a7484aa,32'h5cb0a9dc,32'h76f988da,
    32'h983e5152,32'ha831c66d,32'hb00327c8,32'hbf597fc7,32'hc6e00bf3,32'hd5a79147,32'h06ca6351,32'h14292967,
    32'h27b70a85,32'h2e1b2138,32'h4d2c6dfc,32'h53380d13,32'h650a7354,32'h766a0abb,32'h81c2c92e,32'h92722c85,
    32'ha2bfe8a1,32'ha81a664b,32'hc24b8b70,32'hc76c51a3,32'hd192e819,32'hd6990624,32'hf40e3585,32'h106aa070,
    32'h19a4c116,32'h1e376c08,32'h2748774c,32'h34b0bcb5,32'h391c0cb3,32'h4ed8aa4a,32'h5b9cca4f,32'h682e6ff3,
    32'h748f82ee,32'h78a5636f,32'h84c87814,32'h8cc70208,32'h90befffa,32'ha4506ceb,32'hbef9a3f7,32'hc67178f2
};

// This is the instantiation for stage 1, requiring only 1 sha
simplified_sha256 stg1(.clk(clk), .reset_n(reset_n), .start(begin_stage1), .w(w_block[0]), .hash_in(hash_in), .hash_out(hash1), .hash_done(end_stage1));
// This is a generate for loop to generate the amount of instantions of sha used in stages 2 and 3 depending on the selected mode
// The sha has been changed for the bitcoin hasher, so look at that file for more info. Otherwise, the main thing to know is that the words and hashes are inputted
// directly to the sha module and it receives its data IMMEDIATELY, so that doesn't take up cycles. When it finishes processing, data is IMMEDIATELY available for the arrays in the top module.
// the begin signal is used to begin sha processing, and the end_stage signal is used to signify that the values are ready to be extracted into arrays in the top module
// our bitcoin hasher module
genvar g;
generate
for(g = 0; g < para1; g+=1) begin: hashing_instances
simplified_sha256 hasher(.clk(clk), .reset_n(reset_n), .start(begin_stage2), .w(w_block[g]), .hash_in(hash_in), .hash_out(hash_out[g]), .hash_done(end_stage2[g]));
end
endgenerate

assign mem_clk = clk;
assign mem_addr = cur_addr + offset;
assign mem_we = cur_we;
assign mem_write_data = cur_write_data;

always_ff @(posedge clk, negedge reset_n)
begin
if(!reset_n) begin
state <= IDLE;
end
else begin
state <= next_state;
case(state)
// wait for the start signal, then initialize hashes and address 
IDLE: begin
	if(start) begin
	hash_in[0]<= 32'h6A09E667;
	hash_in[1]<= 32'hBB67AE85;
	hash_in[2]<= 32'h3C6EF372;
	hash_in[3]<= 32'hA54FF53A;
	hash_in[4]<= 32'h510E527F;
	hash_in[5]<= 32'h9B05688C;
	hash_in[6]<= 32'h1F83D9AB;
	hash_in[7]<= 32'h5BE0CD19;
	offset <= 0;
	cur_addr <= message_addr;
	cur_we <= 0;
	i <= 0;
	end
	else begin
	hash_in[0]<= 0;
	hash_in[1]<= 0;
	hash_in[2]<= 0;
	hash_in[3]<= 0;
	hash_in[4]<= 0;
	hash_in[5]<= 0;
	hash_in[6]<= 0;
	hash_in[7]<= 0;
	cur_addr <= 0; 
	offset <= 0;
	cur_we <= 0;
	i <= 0;
	end
end 

//extracts data from memory using same pipelining method used in the first part of this two part project
// each command to receive data from memory takes one cycle to receive that data
READ: begin
if(offset < 1) begin
offset <= offset + 1;
end
else begin
if (offset < 20) begin
offset <= offset + 1;
message[offset-1] <= mem_read_data;
end
else begin
begin_stage1 <= 1;
for(int x = 0; x < 16; x = x + 1) w_block[0][x] <= message[x];
offset <= offset + 1;
end
end                 
end

//stage 11, after the stage ends, and depending on the mode selected. This stage prepares the words to be used in the first part of stage 2
//if mode 1 is selected, or it prepares the words for all 16 nonces at once so that 16 inst.'s instead of 8 process the next stage all at once
STAGE11: begin
	if(end_stage1) begin
	begin_stage1 <= 0;
	begin_stage2 <= 0;
	for(int y = 0; y < 8; y+=1) begin
	hash_in[y] <= hash1[y];
	end
	for(int x = 0; x < para1; x+=1) begin
				  w_block[x][0] <= message[16];
              w_block[x][1] <= message[17];
              w_block[x][2] <= message[18];
              w_block[x][3] <= x;
              w_block[x][4] <= 32'h80000000;
              w_block[x][5] <= 32'h00000000;
              w_block[x][6] <= 32'h00000000;
              w_block[x][7] <= 32'h00000000;
              w_block[x][8] <= 32'h00000000;
              w_block[x][9] <= 32'h00000000;
              w_block[x][10] <= 32'h00000000;
              w_block[x][11] <= 32'h00000000;
              w_block[x][12] <= 32'h00000000;
              w_block[x][13] <= 32'h00000000;
              w_block[x][14] <= 32'h00000000;
              w_block[x][15] <= 32'd640;
	end
	end
	else begin
	begin_stage1 <= 1;
	begin_stage2 <= 0;
	end
end
// First phase of stage 2 (selected at mode 1): when this finishes it prepares the words of nonces 8-15 for the second phase of stage 2.
STAGE21: begin
if(end_stage2[0]) begin
begin_stage2 <= 0;
	for(int y = 0; y < num_nonces/2; y+=1) begin
		for(int z = 0; z < 8; z+=1) begin
			hash_complete[y][z] <= hash_out[y][z];
		end
	end
	for(int x = 0; x < num_nonces/2; x+=1) begin
				  w_block[x][0] <= message[16];
              w_block[x][1] <= message[17];
              w_block[x][2] <= message[18];
              w_block[x][3] <= x+(num_nonces/2);
              w_block[x][4] <= 32'h80000000;
              w_block[x][5] <= 32'h00000000;
              w_block[x][6] <= 32'h00000000;
              w_block[x][7] <= 32'h00000000;
              w_block[x][8] <= 32'h00000000;
              w_block[x][9] <= 32'h00000000;
              w_block[x][10] <= 32'h00000000;
              w_block[x][11] <= 32'h00000000;
              w_block[x][12] <= 32'h00000000;
              w_block[x][13] <= 32'h00000000;
              w_block[x][14] <= 32'h00000000;
              w_block[x][15] <= 32'd640;
	end
end 
else begin
begin_stage2 <= 1;
end
end
// Second phase of stage 2 OR the only phase of stage 2 if mode 0 is selected. At the end of this stage and depending on the mode selected,
// this stage either prepares for the first phase of stage 3, 0-7 nonces, or for stage 3 to be completed in one stage, all nonces at once
// again, this depends on the mode chose 0 vs/ 1
STAGE22: begin
if(end_stage2[0]) begin
begin_stage2 <= 0;
hash_in[0]<= 32'h6A09E667;
hash_in[1]<= 32'hBB67AE85;
hash_in[2]<= 32'h3C6EF372;
hash_in[3]<= 32'hA54FF53A;
hash_in[4]<= 32'h510E527F;
hash_in[5]<= 32'h9B05688C;
hash_in[6]<= 32'h1F83D9AB;
hash_in[7]<= 32'h5BE0CD19;
	for(int y = 0; y < para1; y+=1) begin
		for(int z = 0; z < 8; z+=1) begin
			hash_complete[y+hash_complete_para][z] <= hash_out[y][z];
		end
	end
	for(int x = 0; x < para1; x+=1) begin
			  w_block[x][0] <= (mode == 1) ? hash_complete[x][0]:hash_out[x][0];
			  w_block[x][1] <= (mode == 1) ? hash_complete[x][1]:hash_out[x][1];
			  w_block[x][2] <= (mode == 1) ? hash_complete[x][2]:hash_out[x][2];
			  w_block[x][3] <= (mode == 1) ? hash_complete[x][3]:hash_out[x][3];
			  w_block[x][4] <= (mode == 1) ? hash_complete[x][4]:hash_out[x][4];
			  w_block[x][5] <= (mode == 1) ? hash_complete[x][5]:hash_out[x][5];
			  w_block[x][6] <= (mode == 1) ? hash_complete[x][6]:hash_out[x][6];
			  w_block[x][7] <= (mode == 1) ? hash_complete[x][7]:hash_out[x][7];
			  w_block[x][8] <= 32'h80000000;
			  w_block[x][9] <= 32'h00000000;
			  w_block[x][10] <= 32'h00000000;
			  w_block[x][11] <= 32'h00000000;
			  w_block[x][12] <= 32'h00000000;
			  w_block[x][13] <= 32'h00000000;
			  w_block[x][14] <= 32'h00000000;
			  w_block[x][15] <= 32'd256;	  
	end
end 
else begin
begin_stage2 <= 1;
end
end
// First phase of stage 3. When completed, it prepares for the second phase of stage 3
STAGE31: begin
if(end_stage2[0]) begin
begin_stage2 <= 0;
for(int y = 0; y < num_nonces/2; y+=1) begin
		for(int z = 0; z < 8; z+=1) begin
			hash_complete[y][z] <= hash_out[y][z];
		end
	end
for(int x = 0; x < num_nonces/2; x+=1) begin
			  w_block[x][0] <= hash_complete[x+num_nonces/2][0];
			  w_block[x][1] <= hash_complete[x+num_nonces/2][1];
			  w_block[x][2] <= hash_complete[x+num_nonces/2][2];
			  w_block[x][3] <= hash_complete[x+num_nonces/2][3];
			  w_block[x][4] <= hash_complete[x+num_nonces/2][4];
			  w_block[x][5] <= hash_complete[x+num_nonces/2][5];
			  w_block[x][6] <= hash_complete[x+num_nonces/2][6];
			  w_block[x][7] <= hash_complete[x+num_nonces/2][7];
			  w_block[x][8] <= 32'h80000000;
			  w_block[x][9] <= 32'h00000000;
			  w_block[x][10] <= 32'h00000000;
			  w_block[x][11] <= 32'h00000000;
			  w_block[x][12] <= 32'h00000000;
			  w_block[x][13] <= 32'h00000000;
			  w_block[x][14] <= 32'h00000000;
			  w_block[x][15] <= 32'd256;	  
	end
end
else begin
begin_stage2 <= 1;
end
end
//Second phase of stage 3, or only phase of stage 3 depending on mode selected
STAGE32: begin
if(end_stage2[0]) begin
begin_stage2 <= 0;
offset <= 0;
cur_addr <= output_addr;
for(int y = 0; y < para1; y+=1) begin
		for(int z = 0; z < 8; z+=1) begin
			hash_complete[y+hash_complete_para][z] <= hash_out[y][z];
		end
	end
end
else begin
begin_stage2 <= 1;
end
end
//write state: write to memory. One cycle required to begin writing to memory.
WRITE: begin

if(i == 0) begin
				  cur_we <= 0;
              cur_write_data <= hash_complete[0][0];
              offset <= 0;
				  i+=1;
end
else begin
if(i < num_nonces+1) begin
				  cur_we <= 1;
              cur_write_data <= hash_complete[i-1][0];
              offset <= i-1;
				  i+=1;
end
else begin
					 cur_we <= 0;
                cur_write_data <= 0;
                offset <= 0;
					 i+=1;
end
end 

      
end

endcase 

end 
end 



//always_comb required to determine next state of FSM. Done becomes a 1 when we have completed write and we transition back to idle. so, this is a mealy model.
//note: we only need the end_stage signal from one of the instantiations to know if all the instantiations have completed their processing of hashes and are ready to provide them back to the arrays in the top module.
always_comb begin
case(state)

IDLE: begin
done = 0;
if(start) next_state = READ;
else next_state = IDLE;
end

READ: begin
done = 0;
if(offset < 21) next_state = READ;
else next_state = STAGE11;
end

STAGE11: begin //decides where to go depending on which mode is selected
done = 0;
if(end_stage1 && mode == 1) next_state = STAGE21;
else begin
if(end_stage1 && mode == 0) next_state = STAGE22;
else next_state = STAGE11;
end
end

STAGE21: begin
done = 0;
if(end_stage2[0]) next_state = STAGE22;
else next_state = STAGE21;
end

STAGE22: begin//decides where to go depending on which mode is selected
done = 0;
if(end_stage2[0] && mode == 1) next_state = STAGE31;
else begin 
if(end_stage2[0] && mode == 0) next_state = STAGE32;
else next_state = STAGE22;
end
end

STAGE31: begin
done = 0;
if(end_stage2[0]) next_state = STAGE32;
else next_state = STAGE31;
end

STAGE32: begin
done = 0;
if(end_stage2[0]) next_state = WRITE;
else next_state = STAGE32;
end

WRITE: begin
if(i<= num_nonces+1) begin
next_state = WRITE;
done = 0;
end
else begin
next_state = IDLE;
done = 1;
end
end

endcase

end

endmodule 