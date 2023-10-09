#include <stdio.h>
#include <math.h>
#include "lstm-10-1-module.c"

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
    float Y_h[1][1][1];

    // Call the entry function
    entry(tensor_X, Y_h);

    // Print the result
    printf("Result: %f\n", Y_h[0][0][0]);
    
    
    return 0;
}
