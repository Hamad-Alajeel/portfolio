## Project Details:
This project demonstrates a multi-stage, FPGA-based Bitcoin hashing design using the SHA-256 algorithm. It supports two modes—mode 0 for maximum parallelism (one SHA-256 instantiation per nonce) and mode 1 for reduced area (half as many SHA-256 modules, processed in two phases). The design features eight FSM states (IDLE, READ, STAGE11, STAGE21, STAGE22, STAGE31, STAGE32, and WRITE) to handle memory reads, hashing computations, and final writes back to memory; certain states are bypassed depending on the chosen mode. The top-level module interfaces with a modified SHA-256 core that eliminates superfluous states from the original implementation, adding only a FINISH state to prevent reusing old hashes between stages. Through careful pipelining, parameter-based instantiation, and a well-structured FSM, the project strikes a balance between speed and resource usage while computing Bitcoin-proof-of-work hashes for up to 16 nonces.







