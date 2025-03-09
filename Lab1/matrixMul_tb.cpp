#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
// Define the matrices dimensions
#define lm 6
#define ln 6
#define lp 6
#define m (1 << lm)
#define n (1 << ln)
#define p (1 << lp)

void matrixMul(uint8_t A[n][m], uint8_t B[m][p], uint32_t AB[n][p]);

int main()
{
	uint8_t A[n][m];
	uint8_t B[m][p];
	uint32_t C[n][p];
	uint32_t C_SW[n][p];

	int i, j, k;
	//initialize A
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < m; j++)
		{
			A[i][j] = rand() % 256;
		}
	}
	//initialize B
	for(i = 0; i < m; i++)
	{
		for(j = 0; j < p; j++)
		{
			B[i][j] = rand() % 256;
		}
	}

	//calculate C through the matrixMul function
	matrixMul(A, B, C);

	//compare with software results
	//uint32_t product;
	for(i = 0; i < n; i++)
		{
			for(j = 0; j < p; j++)
			{
				C_SW[i][j] = 0;
				for(k = 0; k < m; k++){
					C_SW[i][j] += A[i][k]*B[k][j];
				}
				//printf("line: %d, column: %d , hw calculation: %d , sw calculation: %d \n", i, j, C[i][j], C_SW[i][j]);
				if(C[i][j] != C_SW[i][j])
				{
					printf("wrong result calculated\nTest failed");
					return 1;
				}
			}
		}
	printf("Test Passed!\n");

	return 0;
}
