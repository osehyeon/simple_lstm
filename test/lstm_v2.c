#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define N_POINTS 50 
#define M_PI 3.14159265358979323846

#define N_IN 10 
#define N_MID 20 
#define N_OUT 1 

float w[4][N_IN][N_MID];
float v[4][N_MID][N_MID];
float b[4][N_MID];

void linspace(double start, double end, int n, double* result) {
    double step = (end - start) / (n - 1);
    for (int i = 0; i < n; i++) {
        result[i] = start + i * step;
    }
}

double random_noise() {
    return ((double) rand() / RAND_MAX) * 0.2 - 0.1; // -0.1부터 0.1 사이의 난수 생성
}


float forward(x, y_prev, c_prev)
{
    
}




int main() {
    // 입력 데이터 생성 
    srand(time(NULL)); 
    double sin_x[N_POINTS];
    double sin_y[N_POINTS];
    linspace(-2 * M_PI, 2 * M_PI, N_POINTS, sin_x);
    for (int i = 0; i < N_POINTS; i++) {
        sin_y[i] = sin(sin_x[i]) + random_noise();
    }

}