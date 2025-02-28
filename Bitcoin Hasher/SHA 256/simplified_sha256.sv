module simplified_sha256 #(parameter integer NUM_OF_WORDS = 20)(
input logic  clk, reset_n, start,
input logic  [15:0] message_addr, output_addr,
output logic done, mem_clk, mem_we,
output logic [15:0] mem_addr,
output logic [31:0] mem_write_data,
input logic [31:0] mem_read_data);
 
// FSM state variables 
enum logic [2:0] {IDLE, READ, BLOCK, COMPUTE, WRITE} state;
 
// Local variables
logic [31:0] w[16];
logic [31:0] message[20];
logic [31:0] h0, h1, h2, h3, h4, h5, h6, h7;
logic [31:0] a, b, c, d, e, f, g, h, optim; // optim is an optimization variable that precomputes k + h for the next cycle in the sha function
logic [ 7:0] i, j; // counter variables, j is used to keep track of which block we are processing
logic [15:0] offset; //offset used in reading/writing from memory
logic [ 7:0] num_blocks;
logic        cur_we;//enable to signal to read or write from memory
logic [15:0] cur_addr;
logic [31:0] cur_write_data;//current data to be written to memory

 
// SHA256 K constants
parameter int k[0:63] = '{
   32'h428a2f98,32'h71374491,32'hb5c0fbcf,32'he9b5dba5,32'h3956c25b,32'h59f111f1,32'h923f82a4,32'hab1c5ed5,
   32'hd807aa98,32'h12835b01,32'h243185be,32'h550c7dc3,32'h72be5d74,32'h80deb1fe,32'h9bdc06a7,32'hc19bf174,
   32'he49b69c1,32'hefbe4786,32'h0fc19dc6,32'h240ca1cc,32'h2de92c6f,32'h4a7484aa,32'h5cb0a9dc,32'h76f988da,
   32'h983e5152,32'ha831c66d,32'hb00327c8,32'hbf597fc7,32'hc6e00bf3,32'hd5a79147,32'h06ca6351,32'h14292967,
   32'h27b70a85,32'h2e1b2138,32'h4d2c6dfc,32'h53380d13,32'h650a7354,32'h766a0abb,32'h81c2c92e,32'h92722c85,
   32'ha2bfe8a1,32'ha81a664b,32'hc24b8b70,32'hc76c51a3,32'hd192e819,32'hd6990624,32'hf40e3585,32'h106aa070,
   32'h19a4c116,32'h1e376c08,32'h2748774c,32'h34b0bcb5,32'h391c0cb3,32'h4ed8aa4a,32'h5b9cca4f,32'h682e6ff3,
   32'h748f82ee,32'h78a5636f,32'h84c87814,32'h8cc70208,32'h90befffa,32'ha4506ceb,32'hbef9a3f7,32'hc67178f2
};
 
 
assign num_blocks = determine_num_blocks(NUM_OF_WORDS); 

 

function logic [15:0] determine_num_blocks(input logic [31:0] size);
 
  determine_num_blocks = 2;
 
endfunction
 
 
// the main sha function. Optim is a variable that precomputes h + k to increase Fmax and performance.
function logic[255:0] sha256_op(input logic [31:0] a, b, c, d, e, f, g,/* w,*/ optim);
logic [31:0] S1, S0, ch, maj, t1, t2; // internal signals
begin
S1 = rightrotate(e, 6) ^ rightrotate(e, 11) ^ rightrotate(e, 25);
ch = (e & f) ^ ((~e) & g);
t1 = ch + S1 + optim;
S0 = rightrotate(a, 2) ^ rightrotate(a, 13) ^ rightrotate(a, 22);
maj = (a & b) ^ (a & c) ^ (b & c);
t2 = maj + S0;
sha256_op = {t1 + t2, a, b, c, d + t1, e, f, g};
end
endfunction

// function used in the creation of new words. Depends on previous words in the 16 element word array, so it can generate words in the word expansion
// for each word index greater than 15.
function logic[31:0] wtnew;
logic [31:0] S1,S0;
begin
S0 = rightrotate(w[1],7) ^ rightrotate(w[1],18) ^ (w[1] >> 3);
S1 = rightrotate(w[14],17) ^ rightrotate(w[14],19) ^ (w[14] >> 10);
wtnew = w[0] + S0 + w[9] + S1;
end
endfunction: wtnew
 
// Generate request to memory
// for reading from memory to get original message
// for writing final computed has value
 
assign mem_clk = clk;
assign mem_addr = cur_addr + offset;
assign mem_we = cur_we;
assign mem_write_data = cur_write_data;
 
function logic [31:0] rightrotate(input logic [31:0] x,
                                   input logic [ 7:0] r);
              rightrotate = (x >> r) | (x << (32 - r));
endfunction
 
 
// SHA-256 FSM 
// Get a BLOCK from the memory, COMPUTE Hash output using SHA256 function
// and write back hash value back to memory
always_ff @(posedge clk, negedge reset_n)
begin
  if (!reset_n) begin
    cur_we <= 1'b0;
    state <= IDLE;
  end 
  else case (state)
    // Initialize hash values h0 to h7 and a to h, other variables and memory we, address offset, etc
    IDLE: begin 
       if(start) begin
       //initial hash values for our design
                           h0 <= 32'h6A09E667;
                           h1 <= 32'hBB67AE85;
                           h2 <= 32'h3C6EF372;
                           h3 <= 32'hA54FF53A;
                           h4 <= 32'h510E527F;
                           h5 <= 32'h9B05688C;
                           h6 <= 32'h1F83D9AB;
                           h7 <= 32'h5BE0CD19;
                           a <= 32'h6a09e667;
                           b <= 32'hbb67ae85;
                           c <= 32'h3c6ef372;
                           d <= 32'ha54ff53a;
                           e <= 32'h510e527f;
                           f <= 32'h9b05688c;
                           g <= 32'h1f83d9ab;
                           h <= 32'h5be0cd19;
                           cur_addr <= message_addr;
                           cur_we <= 0;
                           offset <= 0;
                           i <= 0;
                           j <= 0;
                           state <= READ;
       end
                           else
                           begin
                           h0 <= 0;
                           h1 <= 0;
                           h2 <= 0;
                           h3 <= 0;
                           h4 <= 0;
                           h5 <= 0;
                           h6 <= 0;
                           h7 <= 0;
                           a <= 0;
								  b <= 0;
								  c <= 0;
								  d <= 0;
								  e <= 0;
								  f <= 0;
								  g <= 0;
								  h <= 0;
                           cur_addr <= 0;
                           cur_we <= 0;
                           offset <= 0;
                           i <= 0;
                           j <= 0;
                           state <= IDLE;
                           end
    end
              // begin reading from memory. Each command to receive data from memory requires one extra clock cycle for the data to be avavilable
				  //so, this state pipelines the process of reading from memory efficiently by inputting data available from memory into the message array
				  // at an index of offset - 1.
				  
						READ: begin
						if(offset < 1) begin
						offset <= offset + 1;
						state <= READ;
						end
						else begin
						if(offset < 21) begin
						offset <= offset + 1;
						message[offset-1] <= mem_read_data;
						state <= READ;
						end
						else begin
						offset <= 0;
						state <= BLOCK;
						end
						end
						end
              
    
    BLOCK: begin
              // Fetch message in 512-bit block size
              // For each of 512-bit block initiate hash value computation
				  // if we will process first block, input first 16 words from message array, otherwise for the second block, input last 4 words in addition to padding
				  // which is implemented as per the specification given to us in the instructions
    if(j <  num_blocks) begin
              state <= COMPUTE;
              j <= j + 1;
				  
              if(j == 0) begin
              
               for(int x = 0; x < 16; x = x + 1) w[x] <= message[x];
				  optim <= h + k[0] + message[0]; // pre-computation of optim depends on fist block or second block use message[0] vs message[16]
              end
              
               else begin
              w[0] <= message[16];
              w[1] <= message[17];
              w[2] <= message[18];
              w[3] <= message[19];
              w[4] <= 32'h80000000;
              w[5] <= 32'h00000000;
              w[6] <= 32'h00000000;
              w[7] <= 32'h00000000;
              w[8] <= 32'h00000000;
              w[9] <= 32'h00000000;
              w[10] <= 32'h00000000;
              w[11] <= 32'h00000000;
              w[12] <= 32'h00000000;
              w[13] <= 32'h00000000;
              w[14] <= 32'h00000000;
              w[15] <= 32'd640;
				  optim <= h + k[0] + message[16];// pre-computation of optim depends on fist block or second block use message[0] vs message[16]
              end
              
               end
              else begin
              cur_addr <= output_addr; // when we have finished processing all blocks, we can begin writing so we initialize the current address to the address in memory we want to write to.
              state <= WRITE;
    end
              end
    // For each block compute hash function
    // Go back to BLOCK stage after each block hash computation is completed and if
    // there are still number of message blocks available in memory otherwise
    // move to WRITE stage              

              
               COMPUTE: begin
              if(i < 64) begin
                           for(int m = 0; m < 15; m+=1) w[m] <= w[m+1]; // rotate word values at each cycle
                           w[15] <= wtnew;//create new word because we are expanding the words and processing the sha function all at once
                           {a,b,c,d,e,f,g,h} <= sha256_op(a, b, c, d, e, f, g, optim);
									optim <= k[i+1] + g + w[1];// pre-computation for next cycle
                           i <= i + 1;
                           state <= COMPUTE;
              end
              else begin // when finished with computing the hashes input new hashes in h0 to h7.
                             h0 <= h0 + a;
                             h1 <= h1 + b;
                             h2 <= h2 + c;
                             h3 <= h3 + d;
                             h4 <= h4 + e;
                             h5 <= h5 + f;
                             h6 <= h6 + g;
                             h7 <= h7 + h;
                             a <= h0 + a;
                             b <= h1 + b;
                             c <= h2 + c;
                             d <= h3 + d;
                             e <= h4 + e;
                             f <= h5 + f;
                             g <= h6 + g;
                             h <= h7 + h;
                             i <= 0;
                             state <= BLOCK;
                             end
              end
 
              
               
               
               
               
               
               
    //write back to memory. Requires one clock cycle for data to be available from memory when cur_we is high
    WRITE: begin
 
              if(i <= 9) begin
              case(i)
              0:begin
              cur_we <= 0;
              cur_write_data <= h0;
              offset <= 0;
              end
              1: begin
              cur_we <= 1;
              cur_write_data <= h0;
              offset <= 0;
              end
              2: begin
              cur_we <= 1;
              cur_write_data <= h1;
              offset <= 1;
              end
              3: begin
              cur_we <= 1;
              cur_write_data <= h2;
              offset <= 2;
              end
              4: begin
              cur_we <= 1;
              cur_write_data <= h3;
              offset <= 3;
              end
              5: begin
              cur_we <= 1;
              cur_write_data <= h4;
              offset <= 4;
              end
              6: begin
              cur_we <= 1;
              cur_write_data <= h5;
              offset <= 5;
              end
              7: begin
              cur_we <= 1;
              cur_write_data <= h6;
              offset <= 6;
              end
              8: begin
              cur_we <= 1;
              cur_write_data <= h7;
              offset <= 7;
              end
              default: begin
              cur_we <= 0;
              cur_write_data <= 0;
              offset <= 0;
              end
              endcase
              i <= i + 1;
              end 
              else begin
              state <= IDLE;
              end
 
    end
   endcase
  end
 
// Generate done when SHA256 hash computation has finished and moved to IDLE state
assign done = (state == IDLE);
 
endmodule
