#include <math.h>

typedef struct { 
    float data[1][10];
} Gate; 

typedef struct { 
    float data[1][1][10];
} TensorY; 

float sigmoid(float input);

// cell gate 
 Gate tanh_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], TensorY Y_h)
 {
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;

    Gate gate = {{{0}}}; 

    for(int b=0; b<bs; b++)
    for(int h=0; h<hs; h++) {
        for(int i=0; i<ds; i++) {
            gate.data[b][h] += X[0][b][i]*W[0][h][i];
        }
        for(int k=0; k<hs; k++) {
            gate.data[b][h] += Y_h.data[0][b][k]*R[0][h][k];
        }
        gate.data[b][h] += B[0][h];
        gate.data[b][h] = tanh(gate.data[b][h]);
    }
    return gate; 
}

// input gate, output gate, forget gate 
Gate sigmoid_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], TensorY Y_h)
 {
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;

    Gate gate = {{{0}}}; 

    for(int b=0; b<bs; b++)
    for(int h=0; h<hs; h++) {
        for(int i=0; i<ds; i++) {
            gate.data[b][h] += X[0][b][i]*W[0][h][i];
        }
        for(int k=0; k<hs; k++) {
            gate.data[b][h] += Y_h.data[0][b][k]*R[0][h][k];
        }
        gate.data[b][h] += B[0][h];
        gate.data[b][h] = tanh(gate.data[b][h]);
    }
    return gate; 
}

// A0: forget gate, A1: input gate, A2: cell gate
TensorY cell_state(Gate A0, Gate A1, Gate A2, TensorY Y_c) {
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;

    for( int b=0; b<bs; b++)
    for( int h=0; h<hs; h++) {
        Y_c.data[0][b][h] = Y_c.data[0][b][h]*A0.data[b][h] + A1.data[b][h]*A2.data[b][h];
    }
    return Y_c;
}

// A3: output gate
TensorY lstm_output(Gate A3, TensorY Y_c, TensorY Y_h) {
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;
    
    for( int b=0; b<bs; b++)
      for( int h=0; h<hs; h++) {
         Y_h.data[0][b][h] = A3.data[b][h] * tan_h(Y_c.data[0][b][h]);

      }
      return Y_h;
}

// A1: Reset gate, 
TensorY gru_output(float X[1][1][1], float W[1][10][1], float R[1][10][1], float B[1][10], Gate A0, Gate A1, TensorY Y_h)
{
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;

    Gate gate = {{{0}}}; 

    for(int b=0; b<bs; b++)
    for(int h=0; h<hs; h++) {
        for(int i=0; i<ds; i++) {
            gate.data[b][h] += X[0][b][i]*W[0][h][i];
        }
        gate.data[b][h] += A1.data[b][h]*Y_h.data[0][b][h];
        for(int k=0; k<hs; k++) {
            gate.data[b][h] += Y_h.data[0][b][k]*R[0][h][k];
        }
        gate.data[b][h] += B[0][h];
        gate.data[b][h] = tanh(gate.data[b][h]);
    }
}



float sigmoid(float input) {
    return 1.0f/(1+expf(-input));
}