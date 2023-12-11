#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

/*
static inline void node__fc_Gemm( const float A[1][128], const float B[64][128], const float C[64], float Y[1][64] )
{

	const int M = 1;
	const int K = 128;
	const int N = 64;
	float alpha = 1.0000000000000000000;
	float beta = 1.0000000000000000000;
	float (*C_)[64]  = (float(*)[64])C;
	for( uint32_t r=0; r<M; r++ )
		for( uint32_t c=0; c<N; c++ ) {
			float ABrc = 0;
			for( uint32_t i=0; i<K; i++ ) {
				float B_el = B[c][i];
				ABrc += A[r][i] * B_el;
			}
			float tmp = ABrc * alpha;
			tmp += C_[0][c] * beta;
			Y[r][c] = tmp;
	}
}

static inline void node_StatefulPartitionedCall_dense_MatMul( const float A[1][128], const float B[128][64], float Y[1][64] )
{
	for( uint32_t r=0; r<1; r++ )
		for( uint32_t c=0; c<64; c++ ) {
			Y[r][c] = 0;
			for( uint32_t i=0; i<128; i++ )
				Y[r][c] += A[r][i] * B[i][c];
		}
}
*/

void Gemm_pt(float A[1][128], float B[64][128], float C[64], float Y[1][64])
{
    for(int c=0; c<64; c++ ) {
        float ABrc = 0;
        for(int i=0; i<128; i++ ) {
            ABrc += A[0][i] * B[c][i];
        }
        float tmp = ABrc ;
        tmp += C[c] ;
        Y[0][c] = tmp;
	}
}

void Matmul_tf(float A[1][128], float B[128][64], float Y[1][64])
{
	
    for( uint32_t c=0; c<64; c++ ) { // output_size
        Y[0][c] = 0;
        for( uint32_t i=0; i<128; i++ ) // input_size
            Y[0][c] += A[0][i] * B[i][c];
    }
}

void Matmul_pt(float A[1][128], float B[64][128], float Y[1][64])
{
    for(int c=0; c<64; c++ ) { // output_size 
        Y[0][c]  = 0;
        for(int i=0; i<128; i++ ) { // input_size
            Y[0][c] += A[0][i] * B[c][i];
        }
	}
}

void W_transpose(float src[128][64], float dst[64][128]) {
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 64; ++j) {
            dst[j][i] = src[i][j];
        }
    }
}

int main() {
    float A[1][128];
    float B[128][64];
    float B_transposed[64][128];
    float Y_tf[1][64];
    float Y_pt[1][64];

    // 랜덤 행렬 초기화
    srand(time(NULL));
    for (int i = 0; i < 128; ++i) {
        A[0][i] = (float)rand() / RAND_MAX;
        for (int j = 0; j < 64; ++j) {
            B[i][j] = (float)rand() / RAND_MAX;
        }
    }

    // Matmul_tf 실행
    Matmul_tf(A, B, Y_tf);

    // B 행렬 전치
    W_transpose(B, B_transposed);

    // Matmul_pt 실행
    Matmul_pt(A, B_transposed, Y_pt);

    // 결과 비교
    for (int i = 0; i < 64; ++i) {
        if (fabs(Y_tf[0][i] - Y_pt[0][i]) > 1e-6) {
            printf("Discrepancy found at element %d\n", i);
            return 1;
        }
    }

    printf("The results are consistent.\n");
    return 0;
}