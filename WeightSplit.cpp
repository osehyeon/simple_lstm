#ifndef WEIGHT_SPLIT_H
#define WEIGHT_SPLIT_H

void split_4_tensor_W(float tensor_W[1][512][1], float Wi[1][128][1], float Wo[1][128][1], float Wf[1][128][1], float Wu[1][128][1], int STEPS);
void split_4_tensor_R(float tensor_R[1][512][128], float Ri[1][128][128], float Ro[1][128][128], float Rf[1][128][128], float Ru[1][128][128], int STEPS); 
void split_4_tensor_B(float tensor_B[1][1024], float Bi[1][128], float Bo[1][128], float Bf[1][128], float Bu[1][128], int STEPS);
void split_4_tensor_W(float tensor_W[1][512][1], float Wi[1][128][1], float Wo[1][128][1], float Wf[1][128][1], float Wu[1][128][1], int STEPS) {
// Weights에서 Input Gate, Output Gate, Forget Gate, Update Gate에서 사용할 가중치를 분리하기 위한 함수  
    for (int i = 0; i < 128; i++) {

        if(i >= STEPS) break;

        Wi[0][i][0] = tensor_W[0][i][0]; // Input Gate Weights 
        Wo[0][i][0] = tensor_W[0][STEPS + i][0];// Output Gate Weights 
        Wf[0][i][0] = tensor_W[0][2 * STEPS + i][0]; // Forget Gate Weights 
        Wu[0][i][0] = tensor_W[0][3 * STEPS + i][0]; // Update Gate Weights 
    }
}

void split_4_tensor_R(float tensor_R[1][512][128], float Ri[1][128][128], float Ro[1][128][128], float Rf[1][128][128], float Ru[1][128][128], int STEPS) {
// Recurrence Weights에서 Input Gate, Output Gate, Forget Gate, Update Gate에서 사용할 가중치를 분리하기 위한 함수  
    for (int i = 0; i < 128; i++) {

        if (i >= STEPS) break; 

        for (int j = 0; j < 128; j++) {

            if (j >= STEPS) break;

            Ri[0][i][j] = tensor_R[0][i][j]; // Input Gate Recurrence Weights 
            Ro[0][i][j] = tensor_R[0][STEPS + i][j]; // Output Gate Recurrence Weights
            Rf[0][i][j] = tensor_R[0][2 * STEPS + i][j]; // Forget Gate Recurrence Weights
            Ru[0][i][j] = tensor_R[0][3 * STEPS + i][j]; // Update Gate Recurrence Weights
        }
    }
}

void split_4_tensor_B(float tensor_B[1][1024], float Bi[1][128], float Bo[1][128], float Bf[1][128], float Bu[1][128], int STEPS) {
// Bias에서 Input Gate, Output Gate, Forget Gate, Update Gate에서 사용할 가중치를 분리하기 위한 함수  
    for (int i = 0; i < 128; i++) {

        if (i > STEPS) break;

        Bi[0][i] = tensor_B[0][i] + tensor_B[0][4 * STEPS + i]; // Input Gate Bias 
        Bo[0][i] = tensor_B[0][STEPS + i] + tensor_B[0][5 * STEPS + i]; // Output Gate Bias 
        Bf[0][i] = tensor_B[0][2 * STEPS + i] + tensor_B[0][6 * STEPS + i]; // Forget Gate Bias 
        Bu[0][i] = tensor_B[0][3 * STEPS + i] + tensor_B[0][7 * STEPS + i]; // Update Gate Bias 
    }
}

#endif