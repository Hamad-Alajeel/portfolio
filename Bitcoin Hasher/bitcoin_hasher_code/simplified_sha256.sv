module simplified_sha256 #(parameter integer NUM_OF_WORDS = 20)(
input logic  clk, reset_n, start,
input logic [31:0] w[16],
input logic [31:0] hash_in[8],
output logic [31:0] hash_out[8],
output logic hash_done);
 
// FSM state variables: only 3 states because we receive data IMMEDIATELY from top module
enum logic [2:0] {IDLE, COMPUTE, FINISH} state;
 
 
// Local variables
logic cur_done;
logic [31:0] w_arr[16];
logic [31:0] a, b, c, d, e, f, g, h, optimi;
logic [ 7:0] i;

 
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
 
 assign hash_done = cur_done;

// SHA256 hash round
function logic[255:0] sha256_op(input logic [31:0] a, b, c, d, e, f, g, wi);
logic [31:0] S1, S0, ch, maj, t1, t2; 
begin
S1 = rightrotate(e, 6) ^ rightrotate(e, 11) ^ rightrotate(e, 25);
ch = (e & f) ^ ((~e) & g);
t1 = ch + S1 + wi + optimi;
S0 = rightrotate(a, 2) ^ rightrotate(a, 13) ^ rightrotate(a, 22);
maj = (a & b) ^ (a & c) ^ (b & c);
t2 = maj + S0;
sha256_op = {t1 + t2, a, b, c, d + t1, e, f, g};
end
endfunction
 
function logic [31:0] wik;
	logic [31:0] S1, S0;
	begin
		S0 = rightrotate(w_arr[1], 7) ^ rightrotate(w_arr[1], 18) ^ (w_arr[1] >> 3);
		S1 = rightrotate(w_arr[14], 17) ^ rightrotate(w_arr[14], 19) ^ (w_arr[14] >> 10);
		wik = w_arr[0] + S0 + w_arr[9] + S1;
	end
endfunction
 

 
function logic [31:0] rightrotate(input logic [31:0] x,input logic [ 7:0] r);
              rightrotate = (x >> r) | (x << (32 - r));
endfunction
 
 

always_ff @(posedge clk, negedge reset_n)
begin
  if (!reset_n) begin
    state <= IDLE;
	 cur_done <= 0;
  end 
  else begin
  case (state)
    
    IDLE: begin 
       if(start) begin
									cur_done <= 0; 
                           a <= hash_in[0];
                           b <= hash_in[1];
                           c <= hash_in[2];
                           d <= hash_in[3];
                           e <= hash_in[4];
                           f <= hash_in[5];
                           g <= hash_in[6];
                           h <= hash_in[7];
                           i <= 0;
									
									optimi <= hash_in[7] + k[0];// + w[0];
									for(int x = 0; x < 16; x+=1) begin
										w_arr[x] <=  w[x];
									end
                           state <= COMPUTE;
									end
                           else
                           begin
									cur_done <= 0;
                           a <= 0;
								  b <= 0;
								  c <= 0;
								  d <= 0;
								  e <= 0;
								  f <= 0;
								  g <= 0;
								  h <= 0;
                           i <= 0;
									optimi <= 0;
                           for(int x = 0; x < 16; x+=1) begin
										w_arr[x] <= 0;
									end
                           state <= IDLE;
                           end
    end
              

              //same optimizations used here as in the first part of the project.
               COMPUTE: begin
              if(i < 64) begin
                           
									for(int m = 0; m < 15; m = m + 1) w_arr[m] <= w_arr[m + 1];
                           w_arr[15] <= wik;
                           {a,b,c,d,e,f,g,h} <= sha256_op(a, b, c, d, e, f, g, w_arr[0]/*, optim*/);
									optimi <= k[i+1] + g;
                           i <= i + 1;
                           state <= COMPUTE;
              end
              else begin
                             hash_out[0] <= hash_in[0] + a;
                             hash_out[1] <= hash_in[1] + b;
                             hash_out[2] <= hash_in[2] + c;
                             hash_out[3] <= hash_in[3] + d;
                             hash_out[4] <= hash_in[4] + e;
                             hash_out[5] <= hash_in[5] + f;
                             hash_out[6] <= hash_in[6] + g;
                             hash_out[7] <= hash_in[7] + h;
									  cur_done <= 1;
                             state <= FINISH;
                             end
              end
// finish acts as a buffer to allow the top module to switch the start signal to 0 so that the sha doesn't begin processing incorrect input hashes and words
// the top module changes those depending on the next stage to be completed
    FINISH: begin
		state <= IDLE;
		cur_done <= 0; 
    end
	 
   endcase
	end
  end
 

 
endmodule
