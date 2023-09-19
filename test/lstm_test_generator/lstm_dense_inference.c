#include <stdio.h>
#include <math.h>
#include "help.c"

// Define constants
#define PI 3.14159265358979323846

int main() {
    // Define input tensor
    float tensor_X[10][1][1];
    
    // Fill tensor_X with linspace values between 0 and 2*pi
    float delta = 2.0f * PI / 9.0f;
    for (int i = 0; i < 10; i++) {
        tensor_X[i][0][0] = i * delta;
    }

    // Define output tensor
    float tensor_dense_out[1][128];

    // Call the entry function
    entry(tensor_X, tensor_dense_out);

    // Print the result
    for(int i=0; i<128; i++) {
        printf("Result: %f\n", tensor_dense_out[0][i]);
    }
    
    return 0;
}
