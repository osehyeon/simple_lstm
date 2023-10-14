#include <math.h>
#include <string.h>
#include <stdio.h>

typedef struct { 
    float data[1][10];
} Gate; 

typedef struct { 
    float data[1][1][10];
} TensorY; 

float sigmoid(float input);
float tan_h(float input);
Gate input_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], TensorY Y_h);
Gate output_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], TensorY Y_h);
Gate forget_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], TensorY Y_h);
Gate update_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], TensorY Y_h);
TensorY cell_state(Gate ft, Gate it, Gate ct, TensorY Y_c);
TensorY output(Gate ot, TensorY Y_c, TensorY Y_h);

TensorY entry(float tensor_X[10][1][1],
            float Wi[1][10][1], float Wo[1][10][1], float Wf[1][10][1], float Wu[1][10][1],
            float Ri[1][10][10], float Ro[1][10][10], float Rf[1][10][10], float Ru[1][10][10],
            float Bi[1][10], float Bo[1][10], float Bf[1][10], float Bu[1][10],
            float temp_Bi[1][10], float temp_Bo[1][10], float temp_Bf[1][10], float temp_Bu[1][10]) {
                
    TensorY Y_h = {{{0}}};
    TensorY Y_c = {{{0}}};
    Gate it = {{0}};
    Gate ot = {{0}};
    Gate ft = {{0}};
    Gate ct = {{0}};
    
    float X[1][1][1]= {{{0}}};

    for(int i = 0; i < 10; i++)
    {
        Bi[0][i] += temp_Bi[0][i];
        Bo[0][i] += temp_Bo[0][i];
        Bf[0][i] += temp_Bf[0][i];
        Bu[0][i] += temp_Bu[0][i];
    }

    for(int s=0; s<10; s++) {
        X[0][0][0] = tensor_X[s][0][0];
        it = input_gate(X, Wi, Ri, Bi, Y_h);
        ot = output_gate(X, Wo, Ro, Bo, Y_h);
        ft = forget_gate(X, Wf, Rf, Bf, Y_h);
        ct = update_gate(X, Wu, Ru, Bu, Y_h);
        Y_c = cell_state(ft, it, ct, Y_c);
        Y_h = output(ot, Y_c, Y_h);
        /*
        if(s == 0)
        {
            for(int i=0; i<10; i++)
            {
                printf("it : %f \n", it.data[0][i]);
            }
            for(int i=0; i<10; i++)
            {
                printf("ot : %f \n", ot.data[0][i]);
            }
            for(int i=0; i<10; i++)
            {
                printf("ft : %f \n", ft.data[0][i]);
            }
            for(int i=0; i<10; i++)
            {
                printf("ct : %f \n", ct.data[0][i]);
            }
            for(int i=0; i<10; i++)
            {
                printf("Y_c : %f \n", Y_c.data[0][0][i]);
            }
            for(int i=0; i<10; i++)
            {
                printf("Y_h : %f \n", Y_h.data[0][0][i]);
            }
        }
        */
    }

    return Y_h;
}

float sigmoid(float input) {
    return 1.0f/(1+expf(-input));
}

float tan_h(float input) {
    return tanh(input);
}

Gate input_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], TensorY Y_h){
    
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;

    Gate it = {{{0}}}; 

    for(int b=0; b<bs; b++)
    for(int h=0; h<hs; h++) {
        for(int i=0; i<ds; i++) {
            it.data[b][h] += X[0][b][i]*W[0][h][i];
        }
        for(int k=0; k<hs; k++) {
            it.data[b][h] += Y_h.data[0][b][k]*R[0][h][k];
        }
        it.data[b][h] += B[0][h];
        it.data[b][h] = sigmoid(it.data[b][h]);
    }

    return it; 
}

Gate output_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], TensorY Y_h){
    
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;

    Gate ot = {{{0}}}; 

    for(int b=0; b<bs; b++)
    for(int h=0; h<hs; h++) {
        for(int i=0; i<ds; i++) {
            ot.data[b][h] += X[0][b][i]*W[0][h][i];
        }
        for(int k=0; k<hs; k++) {
            ot.data[b][h] += Y_h.data[0][b][k]*R[0][h][k];
        }
        ot.data[b][h] += B[0][h];
        ot.data[b][h] = sigmoid(ot.data[b][h]);
    }

    return ot; 
}

Gate forget_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], TensorY Y_h){
    
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;

    Gate ft = {{{0}}};  

    for(int b=0; b<bs; b++)
    for(int h=0; h<hs; h++) {
        for(int i=0; i<ds; i++) {
            ft.data[b][h] += X[0][b][i]*W[0][h][i];
        }
        for(int k=0; k<hs; k++) {
            ft.data[b][h] += Y_h.data[0][b][k]*R[0][h][k];
        }
        ft.data[b][h] += B[0][h];
        ft.data[b][h] = sigmoid(ft.data[b][h]);
    }
    return ft;
}


Gate update_gate(float X[1][1][1], float W[1][10][1], float R[1][10][10], float B[1][10], TensorY Y_h){
    
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;

    Gate ct = {{{0}}}; 

    for(int b=0; b<bs; b++)
    for(int h=0; h<hs; h++) {
        ct.data[b][h]= 0;
        for(int i=0; i<ds; i++) {
            ct.data[b][h] += X[0][b][i]*W[0][h][i];
        }
        for(int k=0; k<hs; k++) {
            ct.data[b][h] += Y_h.data[0][b][k]*R[0][h][k];
        }
        ct.data[b][h] += B[0][h];
        ct.data[b][h] = tan_h(ct.data[b][h]);
    }

    return ct; 
}

TensorY cell_state(Gate ft, Gate it, Gate ct, TensorY Y_c) {
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;

    for( int b=0; b<bs; b++)
    for( int h=0; h<hs; h++) {
        Y_c.data[0][b][h] = Y_c.data[0][b][h]*ft.data[b][h] + it.data[b][h]*ct.data[b][h];
    }
    return Y_c;
}

TensorY output(Gate ot, TensorY Y_c, TensorY Y_h) {
    const int hs = 10;
    const int ds = 1;
    const int bs = 1;
    
    for( int b=0; b<bs; b++)
      for( int h=0; h<hs; h++) {
         Y_h.data[0][b][h] = ot.data[b][h] * tan_h(Y_c.data[0][b][h]);

      }
      return Y_h;
}