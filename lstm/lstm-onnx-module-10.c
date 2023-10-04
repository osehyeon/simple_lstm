#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "parameter.c"


float sigmoid(float input) {
    return 1.0f/(1+expf(-input));
}

float tan_h(float input) {
    return tanh(input);
}

void forget_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], float Y_h[1][1][10], float ft[1][10]){
    
    const int hs = 10;
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

void input_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], float Y_h[1][1][10], float it[1][10]){
    
    const int hs = 10;
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

void update_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], float Y_h[1][1][10], float ct[1][10]){
    
    const int hs = 10;
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

void output_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], float Y_h[1][1][10], float ot[1][10]){
    
    const int hs = 10;
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

void cell_state(float ft[1][10], float it[1][10], float ct[1][10], float Y_c[1][1][10]) {
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;

    for( int b=0; b<bs; b++)
    for( int h=0; h<hs; h++) {
        Y_c[0][b][h] = Y_c[0][b][h]*ft[b][h] + it[b][h]*ct[b][h];
    }
}

void output(float ot[1][10],  float Y_h[1][1][10], float Y_c[1][1][10]) {
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;
    
    for( int b=0; b<bs; b++)
      for( int h=0; h<hs; h++) {
         Y_h[0][b][h] = ot[b][h] * tan_h(Y_c[0][b][h]);
      }
}

void entry(const float tensor_X[10][1][1], float tensor_Y_h[1][1][10]) {
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

    float Wf[1][10][1];
    float Wi[1][10][1];
    float Wu[1][10][1];
    float Wo[1][10][1];

    memcpy(Wi, &tensor_W[0][0], sizeof(float) * 10);
    memcpy(Wo, &tensor_W[0][10], sizeof(float) * 10);
    memcpy(Wf, &tensor_W[0][20], sizeof(float) * 10);
    memcpy(Wu, &tensor_W[0][30], sizeof(float) * 10);

    float Rf[1][10][10];
    float Ri[1][10][10];
    float Ru[1][10][10];
    float Ro[1][10][10];

    memcpy(Ri, &tensor_R[0][0], sizeof(float) * 10 * 10);
    memcpy(Ro, &tensor_R[0][10], sizeof(float) * 10 * 10);
    memcpy(Rf, &tensor_R[0][20], sizeof(float) * 10 * 10);
    memcpy(Ru, &tensor_R[0][30], sizeof(float) * 10 * 10);

    float Bf[1][10];
    float Bi[1][10];
    float Bu[1][10];
    float Bo[1][10];
    float temp_B[1][10];

    memcpy(Bi, &tensor_B[0][0], sizeof(float) * 10);
    memcpy(temp_B, &tensor_B[0][40], sizeof(float) * 10);
    for(int i = 0; i < 10; i++) {
        Bi[0][i] += temp_B[0][i];
    }
    memcpy(Bo, &tensor_B[0][10], sizeof(float) * 10);
    memcpy(temp_B, &tensor_B[0][50], sizeof(float) * 10);
    for(int i = 0; i < 10; i++) {
        Bo[0][i] += temp_B[0][i];
    }
    memcpy(Bf, &tensor_B[0][20], sizeof(float) * 10);
    memcpy(temp_B, &tensor_B[0][60], sizeof(float) * 10);
    for(int i = 0; i < 10; i++) {
        Bf[0][i] += temp_B[0][i];
    }
    memcpy(Bu, &tensor_B[0][30], sizeof(float) * 10);
    memcpy(temp_B, &tensor_B[0][70], sizeof(float) * 10);
    for(int i = 0; i < 10; i++) {
        Bu[0][i] += temp_B[0][i];
    }
    
    float ft[1][10] = {0};
    float it[1][10] = {0};
    float ct[1][10] = {0};
    float ot[1][10] = {0};

    for( int s=0; s<10; s++) {
        memcpy(X, &tensor_X[s], sizeof(float) * 1 * 1);

        forget_gate(X, Wf, Rf, Bf, tensor_Y_h, ft);
        input_gate(X, Wi, Ri, Bi, tensor_Y_h, it);
        update_gate(X, Wu, Ru, Bu, tensor_Y_h, ct);
        output_gate(X, Wo, Ro, Bo, tensor_Y_h, ot);
        cell_state(ft, it, ct, tensor_Y_c);
        output(ot, tensor_Y_h, tensor_Y_c);
    }
    
}