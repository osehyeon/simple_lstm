//#include <float.h>
//#include <math.h>
//#include <stdbool.h>
//#include <stdint.h>
//#include <string.h>
#include "parameter.c"


float sigmoid(float input) {
    return 1.0f/(1+expf(-input));
}

float tan_h(float input) {
    return tanh(input);
}

void forget_gate(float X[1][1][1], float W[1][128][1], float R[1][128][128], float B[1][128], float Y_h[1][1][128], float ft[1][128]){
    
    const int hs = 128;
    const int ds = 1;
    const int bs = 1;

    memset(ft, 0, sizeof(*ft));

    for(int b=0; b<bs; b++)
    for(int h=0; h<hs; h++) {
        ft[b][h]= 0;
        for(int i=0; i<ds; i++) {
            ft[b][h] += X[0][b][i]*W[0][h][i];
        }
        for(int k=0; k<hs; k++) {
            ft[b][h] += Y_h[0][b][k]*R[0][h][k];
        }
        ft[b][h] += B[0][h];
        ft[b][h] = sigmoid(ft[b][h]);
    }
}

void input_gate(float X[1][1][1], float W[1][128][1], float R[1][128][128], float B[1][128], float Y_h[1][1][128], float it[1][128]){
    
    const int hs = 128;
    const int ds = 1;
    const int bs = 1;

    memset(it, 0, sizeof(*it));

    for(int b=0; b<bs; b++)
    for(int h=0; h<hs; h++) {
        it[b][h]= 0;
        for(int i=0; i<ds; i++) {
            it[b][h] += X[0][b][i]*W[0][h][i];
        }
        for(int k=0; k<hs; k++) {
            it[b][h] += Y_h[0][b][k]*R[0][h][k];
        }
        it[b][h] += B[0][h];
        it[b][h] = sigmoid(it[b][h]);
    }
}

void update_gate(float X[1][1][1], float W[1][128][1], float R[1][128][128], float B[1][128], float Y_h[1][1][128], float ct[1][128]){
    
    const int hs = 128;
   const int ds = 1;
   const int bs = 1;

    memset(ct, 0, sizeof(*ct));

    for(int b=0; b<bs; b++)
    for(int h=0; h<hs; h++) {
        ct[b][h]= 0;
        for(int i=0; i<ds; i++) {
            ct[b][h] += X[0][b][i]*W[0][h][i];
        }
        for(int k=0; k<hs; k++) {
            ct[b][h] += Y_h[0][b][k]*R[0][h][k];
        }
        ct[b][h] += B[0][h];
        ct[b][h] = tan_h(ct[b][h]);
    }
}

void output_gate(float X[1][1][1], float W[1][128][1], float R[1][128][128], float B[1][128], float Y_h[1][1][128], float ot[1][128]){
    
    const int hs = 128;
    const int ds = 1;
    const int bs = 1;

    memset(ot, 0, sizeof(*ot));

    for(int b=0; b<bs; b++)
    for(int h=0; h<hs; h++) {
        ot[b][h]= 0;
        for(int i=0; i<ds; i++) {
            ot[b][h] += X[0][b][i]*W[0][h][i];
        }
        for(int k=0; k<hs; k++) {
            ot[b][h] += Y_h[0][b][k]*R[0][h][k];
        }
        ot[b][h] += B[0][h];
        ot[b][h] = sigmoid(ot[b][h]);
    }
}

void cell_state(float ft[1][128], float it[1][128], float ct[1][128], float Y_c[1][1][128]) {
    const int hs = 128;
    const int ds = 1;
    const int bs = 1;

    for( int b=0; b<bs; b++)
    for( int h=0; h<hs; h++) {
        Y_c[0][b][h] = Y_c[0][b][h]*ft[b][h] + it[b][h]*ct[b][h];
    }
}

void output(float ot[1][128],  float Y_h[1][1][128], float Y_c[1][1][128]) {
    const int hs = 128;
    const int ds = 1;
    const int bs = 1;
    
    for( int b=0; b<bs; b++)
      for( int h=0; h<hs; h++) {
         Y_h[0][b][h] = ot[b][h] * tan_h(Y_c[0][b][h]);
      }
}

static inline void Reshape( const float data[1][1][128], float reshaped[1][128] )
{
	/*Reshape*/
	float *data_ptr = (float*)data;
	float *reshaped_ptr = (float*)reshaped;
	for( uint32_t i=0; i<128; i++ )
		reshaped_ptr[i] = data_ptr[i];
}


void entry(const float tensor_X[10][1][1], float tensor_reshape_last[1][128]) {
/*    
    int hs = 128;
   int ds = 1;
   int bs = 1;
    int iidx = 0;
   int oidx = hs;
   int fidx = 2*hs;
   int cidx = 3*hs;
*/

    float X[1][1][1];

    float Wf[1][128][1];
    float Wi[1][128][1];
    float Wu[1][128][1];
    float Wo[1][128][1];

    memcpy(Wi, &tensor_W[0][0], sizeof(float) * 128);
    memcpy(Wo, &tensor_W[0][128], sizeof(float) * 128);
    memcpy(Wf, &tensor_W[0][256], sizeof(float) * 128);
    memcpy(Wu, &tensor_W[0][384], sizeof(float) * 128);

    float Rf[1][128][128];
    float Ri[1][128][128];
    float Ru[1][128][128];
    float Ro[1][128][128];

    memcpy(Ri, &tensor_R[0][0], sizeof(float) * 128 * 128);
    memcpy(Ro, &tensor_R[0][128], sizeof(float) * 128 * 128);
    memcpy(Rf, &tensor_R[0][256], sizeof(float) * 128 * 128);
    memcpy(Ru, &tensor_R[0][384], sizeof(float) * 128 * 128);

    float Bf[1][128];
    float Bi[1][128];
    float Bu[1][128];
    float Bo[1][128];
    float temp_B[1][128];

    memcpy(Bi, &tensor_B[0][0], sizeof(float) * 128);
    memcpy(temp_B, &tensor_B[0][512], sizeof(float) * 128);
    for(int i = 0; i < 128; i++) {
        Bi[0][i] += temp_B[0][i];
    }
    memcpy(Bo, &tensor_B[0][128], sizeof(float) * 128);
    memcpy(temp_B, &tensor_B[0][640], sizeof(float) * 128);
    for(int i = 0; i < 128; i++) {
        Bo[0][i] += temp_B[0][i];
    }
    memcpy(Bf, &tensor_B[0][256], sizeof(float) * 128);
    memcpy(temp_B, &tensor_B[0][768], sizeof(float) * 128);
    for(int i = 0; i < 128; i++) {
        Bf[0][i] += temp_B[0][i];
    }
    memcpy(Bu, &tensor_B[0][384], sizeof(float) * 128);
    memcpy(temp_B, &tensor_B[0][896], sizeof(float) * 128);
    for(int i = 0; i < 128; i++) {
        Bu[0][i] += temp_B[0][i];
    }
    
    float ft[1][128] = {0};
    float it[1][128] = {0};
    float ct[1][128] = {0};
    float ot[1][128] = {0};

    for( int s=0; s<10; s++) {
        memcpy(X, &tensor_X[s], sizeof(float) * 1 * 1);

        forget_gate(X, Wf, Rf, Bf, tensor_Y_h, ft);
        input_gate(X, Wi, Ri, Bi, tensor_Y_h, it);
        update_gate(X, Wu, Ru, Bu, tensor_Y_h, ct);
        output_gate(X, Wo, Ro, Bo, tensor_Y_h, ot);
        cell_state(ft, it, ct, tensor_Y_c);
        output(ot, tensor_Y_h, tensor_Y_c);
    }

    Reshape(tensor_Y_h, tensor_reshape_last);
    
}