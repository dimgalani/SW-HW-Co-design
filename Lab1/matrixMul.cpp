/*
 *
 * This file contains 2 implementations of the matrixMul function: one using the hls unroll pragma (1st function)
 * and one using the hls pipeline pragma (2nd function)
 *
 *
 * */

#include <stdio.h>
#include <stdint.h>
#define lm 6
#define ln 6
#define lp 6
#define m (1 << lm)
#define n (1 << ln)
#define p (1 << lp)
#define UNROLL_FACTOR 64


//#### ACCELERATION USING UNROLL ####

void matrixMul(uint8_t A[n][m], uint8_t B[m][p], uint32_t AB[n][p]){
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
#pragma HLS ARRAY_PARTITION variable=AB complete dim=2
int i,j,k;
    loop1: for (i = 0; i < n; i++){
        loop2: for (j = 0; j < p; j++){
            int product = 0;
            loop3: for (k = 0; k < m; k++){
#pragma HLS UNROLL factor = UNROLL_FACTOR
                product += A[i][k] * B[k][j];
            }
            AB[i][j] = product;
        }
    }
}

/*
//#### ACCELERATION USING PIPELINE ####

void matrixMul(uint8_t A[n][m], uint8_t B[m][p], uint32_t AB[n][p]){
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
#pragma HLS ARRAY_PARTITION variable=AB complete dim=2
int i,j,k;
    loop1: for (i = 0; i < n; i++){
        loop2: for (j = 0; j < p; j++){
#pragma HLS PIPELINE II=1
            int product = 0;
            loop3: for (k = 0; k < m; k++){
                product += A[i][k] * B[k][j];
            }
            AB[i][j] = product;
        }
    }
}
*/


