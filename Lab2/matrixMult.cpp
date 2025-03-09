# include <stdint.h>
#define BUFFER_SIZE 1024
#define DATA_SIZE 4096

// TRIpCOUnT identifier
const unsigned int c_len = DATA_SIZE / BUFFER_SIZE;
const unsigned int c_size = BUFFER_SIZE;




// Defining the matricies dimensions
#define lm 4
#define ln 4
#define lp 4
#define m (1 << lm)
#define n (1 << ln)
#define p (1 << lp)


extern "C" {
// void matmult(int32_t A[n][m], int32_t B[m][p], int64_t AB[n][p]) {
void vadd(int32_t *A, // Read-Only Vector 1
		 int32_t *B, // Read-Only Vector 2
		 int64_t *AB,     // Output Result
          int size                 // Size in integer
          ) {
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = AB offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = AB bundle = control
#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    int32_t A_buffer[n][m];
    int32_t B_buffer[m][p];
    int64_t AB_buffer[n][p];

#pragma HLS ARRAY_PARTITION variable=A_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=AB_buffer complete dim=2

int i, j, k;
    // Initialize the output matrix
    init:
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
// #pragma HLS PIPELINE II = 1
#pragma HLS UNROLL factor = 16
            AB_buffer[i][j] = 0;
        }
    }

    // Read matrix A into local buffer
    read_A:
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
#pragma HLS PIPELINE II = 1
            A_buffer[i][j] = A[i*m + j];
        }
    }

    // Read matrix B into local buffer
    read_B:
    for (i = 0; i < m; i++) {
        for (j = 0; j < p; j++) {
#pragma HLS PIPELINE II = 1
            B_buffer[i][j] = B[i*p+j];
        }
    }


    // perform matrix multiplication
    compute:
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
//#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL factor = 16
       	 for (k = 0; k < m; k++) {
       		 AB_buffer[i][j] += A_buffer[i][k] * B_buffer[k][j];
            }
        }
    }
//  AB[0] = 7;
//  AB[1] = 10;
//  AB[2] = 15;
//  AB[3] = 22;
    // Write the result matrix C to global memory
    write_C:
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
#pragma HLS PIPELINE II = 1
       	 AB[i*p+j] = AB_buffer[i][j];
        }
    }
}
}




