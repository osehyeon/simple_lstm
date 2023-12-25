#include <math.h>

typedef struct { 
    float data[1][1][128];
} Tensor; 


Tensor compute_gate(float X[1][1][1], float W[1][128][1], float R[1][128][128], float B[1][128], Tensor Y_h, int STEPS)
// gate 연산 모듈 
{
    const int hs = 128;

    Tensor gate = {{{0}}};
    for(int h=0; h<hs; h++)
    {   
        if(h >= STEPS) break;

        gate.data[0][0][h] += X[0][0][0] * W[0][h][0];
        
        for(int k=0; k<hs; k++)
        {
            if (k >= STEPS) break;

            gate.data[0][0][h] += Y_h.data[0][0][k] * R[0][h][k];
        }
        gate.data[0][0][h] += B[0][h];
    }
    return gate;
}

Tensor compute_gate_column(float X[1][1][1], float W[1][128][1], float R[1][128][128], float B[1][128], Tensor Y_h, int STEPS)
// gate 연산 모듈 
{
    const int hs = 128;

    Tensor gate = {{{0}}};
    for(int h=0; h<hs; h++)
    {   
        if(h >= STEPS) break;

        gate.data[0][0][h] += X[0][0][0] * W[0][h][0];
        
        for(int k=0; k<hs; k++)
        {
            if (k >= STEPS) break;

            gate.data[0][0][h] += Y_h.data[0][0][k] * R[0][k][h];
        }
        gate.data[0][0][h] += B[0][h];
    }
    return gate;
}

Tensor compute_sigmoid(Tensor gate, int STEPS) 
// sigmoid 연산 모듈 
{
    const int hs = 128;

    for(int h=0; h<hs; h++)
    {
        if (h >= STEPS) break;

        gate.data[0][0][h] = 1.0f / (1.0f + expf(-gate.data[0][0][h]));
    }
    
    return gate;
}

Tensor compute_relu(Tensor gate, int STEPS) 
// relu 연산 모듈 
{
    const int hs = 128;

    for(int h=0; h<hs; h++)
    {
        if (h >= STEPS) break;

        gate.data[0][0][h] = (gate.data[0][0][h] > 0) ? gate.data[0][0][h] : 0;
    }
    
    return gate;
}

Tensor compute_tanh(Tensor gate, int STEPS) 
// tanh 연산 모듈 
{
    const int hs = 128;

    for(int h=0; h<hs; h++)
    {
        if (h >= STEPS) break;

        gate.data[0][0][h] = tanh(gate.data[0][0][h]);
    }
    
    return gate;
}

Tensor compute_cell_state(Tensor A0, Tensor A1, Tensor A2, Tensor Y_c, int STEPS)
// cell state 연산 모듈 
{
    const int hs = 128;
    
    for(int h=0; h<hs; h++)
    {
        if (h >= STEPS) break;

        Y_c.data[0][0][h] = Y_c.data[0][0][h] * A0.data[0][0][h] + A1.data[0][0][h] * A2.data[0][0][h];
    }
    return Y_c; 
}

Tensor compute_hidden_state(Tensor A3, Tensor Y_c, Tensor Y_h, int STEPS)
// hidden state 연산 모듈 
{
    const int hs = 128;

    for(int h=0; h<hs; h++) {
        if (h >= STEPS) break;

        Y_h.data[0][0][h] = A3.data[0][0][h] * Y_c.data[0][0][h];
    }
    return Y_h;
}

Tensor Dense(Tensor X, float W[1][128][128], float B[1][128], int I_STEPS, int O_STEPS)
{
    Tensor dense = {{{0}}};

    for(int o=0; o<128; o++ ) { 
        if (o >= O_STEPS) break;
        for(int i=0; i<128; i++ )  {
            if (i >= I_STEPS) break;
            dense.data[0][0][o] += X.data[0][0][i] * W[0][i][o];
        }
        dense.data[0][0][o] += B[0][o];
    }

    return dense;
}
