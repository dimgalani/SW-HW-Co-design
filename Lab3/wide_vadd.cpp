#include <ap_int.h>
#include <stdio.h>
#include <string.h>

#define L_M 4
#define L_N 4
#define L_P 4
#define M (1 << L_M)
#define N (1 << L_N)
#define P (1 << L_P)

#define DATAWIDTH 512        // Each element is 512 bits wide.
#define VECTOR_SIZE (DATAWIDTH / 32) // 16 integers per vector (512/32).
#define BUFFER_SIZE 16
#define BITS_PER_ELEMENT 32   // Each element is 32 bits wide.

typedef ap_uint<DATAWIDTH> uint512_dt;

extern "C"
{
    void vadd(
        const uint512_dt *A, // Read-Only Matrix A
        const uint512_dt *B, // Read-Only Matrix B
        uint512_dt *C,        // Output Matrix C
		int size
    )
    {
#pragma HLS INTERFACE m_axi port = A bundle = gmem
#pragma HLS INTERFACE m_axi port = B bundle = gmem1
#pragma HLS INTERFACE m_axi port = C bundle = gmem2
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = C bundle = control
#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

        // Local buffers to store entire matrices
        uint512_dt a_local[BUFFER_SIZE]; // Buffer for matrix A
        uint512_dt b_local[BUFFER_SIZE]; // Buffer for matrix B
        uint512_dt c_local[2 * BUFFER_SIZE]; // Buffer for the resulting row of C

        // Load entire matrices into local buffers?
        for (int i = 0; i < BUFFER_SIZE; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16
//#pragma HLS UNROLL factor=16
            a_local[i] = A[i];
            b_local[i] = B[i];
        }

        // Perform matrix multiplication
        // Itterate over each row of A and column of B
        for (int rowElement = 0; rowElement < BUFFER_SIZE; rowElement++) {  // For each row of A - each row is one element -> contains 16 integers
//#pragma HLS pipeline II=1
//#pragma HLS UNROLL factor=16
        	uint512_dt a_chunk = a_local[rowElement]; // Load one row of A
            for (int columnElement = 0; columnElement < BUFFER_SIZE; columnElement++) { // For each column of B - each column is one element -> contains 16 integers
//#pragma HLS pipeline
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16
#pragma HLS UNROLL factor=16
            uint512_dt b_chunk = b_local[columnElement]; // Load one column of B
            ap_uint<64> sum = 0; // Initialize the dot product to zero
            // Compute dot product of row from A and column from B
                for (int j = 0; j < VECTOR_SIZE; j++) { // For each integer in the row of A and column of B
#pragma HLS UNROLL factor=16
                    ap_uint<32> a_elem = a_chunk.range(32 * (j + 1) - 1, 32 * j); // Extract the j-th integer from the row of A
                    ap_uint<32> b_elem = b_chunk.range(32 * (j + 1) - 1, 32 * j); // Extract the j-th integer from the column of B
                    ap_uint<64> prod = a_elem * b_elem; //Product
                    sum += prod;
                }

                // Store the 64-bit dot product into the 1D c_local
                // 512/64 = 8 integers per 512-bit word
                // One element of C can store 8 integers of 64 bits
                int flat_index = rowElement * BUFFER_SIZE + columnElement; // the index of the integer (0-255)
                // flat_index / 8 chooses one of the 32 512-bit elements of C
                // 1st till 8th result in c_local[0] since flat_index = (0 * 16 + 0)/8 = (0 * 16 + 7)/8 = 0
                // so the first 8 results are stored in c_local[0]
                c_local[flat_index / 8].range(64 * ((flat_index % 8) + 1) - 1, 64 * (flat_index % 8)) = sum;
                // the flat_index % 8 identifies which 64-bit slot (out of 8) within the chosen uint512_t holds the value
        }
        }

        // Write the computed row of C to global memory
        for (int i = 0; i < 2 * BUFFER_SIZE; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 32
#pragma HLS PIPELINE II=1
//#pragma HLS UNROLL factor=32
            C[i] = c_local[i];
        }
    }
}