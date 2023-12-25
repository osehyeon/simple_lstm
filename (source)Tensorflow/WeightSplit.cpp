#ifndef WEIGHT_SPLIT_H
#define WEIGHT_SPLIT_H

void split_4_tensor_W(float tensor_W[1][512][1], float Wi[1][128][1], float Wo[1][128][1], float Wf[1][128][1], float Wu[1][128][1], int STEPS);
void split_4_tensor_R(float tensor_R[1][128][512], float Ri[1][128][128], float Ro[1][128][128], float Rf[1][128][128], float Ru[1][128][128], int STEPS); 
void split_4_tensor_B(float tensor_B[1][512], float Bi[1][128], float Bo[1][128], float Bf[1][128], float Bu[1][128], int STEPS);

void split_4_tensor_W(float tensor_W[1][512][1], float Wi[1][128][1], float Wo[1][128][1], float Wf[1][128][1], float Wu[1][128][1], int STEPS) {
// Weights에서 Input Gate, Output Gate, Forget Gate, Update Gate에서 사용할 가중치를 분리하기 위한 함수  
    for (int i = 0; i < 128; i++) {

        if(i >= STEPS) break;

        Wi[0][i][0] = tensor_W[0][i][0]; // Input Gate Weights 
        Wf[0][i][0] = tensor_W[0][STEPS + i][0];// Output Gate Weights 
        Wu[0][i][0] = tensor_W[0][2 * STEPS + i][0]; // Forget Gate Weights 
        Wo[0][i][0] = tensor_W[0][3 * STEPS + i][0]; // Update Gate Weights 
    }
}

void split_4_tensor_R(float tensor_R[1][128][512], float Ri[1][128][128], float Ro[1][128][128], float Rf[1][128][128], float Ru[1][128][128], int STEPS) {
// Recurrence Weights에서 Input Gate, Output Gate, Forget Gate, Update Gate에서 사용할 가중치를 분리하기 위한 함수  
    
    float Ri_[1][128][128] = {{{0}}};
    float Ro_[1][128][128] = {{{0}}};
    float Rf_[1][128][128] = {{{0}}};
    float Ru_[1][128][128] = {{{0}}};

    for (int i = 0; i < 128; i++) {
        if (i >= STEPS) break; 
        for (int j = 0; j < 128; j++) {
            if (j >= STEPS) break;
            Ri_[0][i][j] = tensor_R[0][i][j]; // Input Gate Recurrence Weights 
            Rf_[0][i][j] = tensor_R[0][i][STEPS + j]; // Output Gate Recurrence Weights
            Ru_[0][i][j] = tensor_R[0][i][2 * STEPS + j]; // Forget Gate Recurrence Weights
            Ro_[0][i][j] = tensor_R[0][i][3 * STEPS + j]; // Update Gate Recurrence Weights
        }
    }
    
    for(int i=0; i<128; i++) {
        if (i >= STEPS) break; 
        for (int j=0; j<128; j++) {
            if (j >= STEPS) break;
            Ri[0][j][i] = Ri_[0][i][j];
            Rf[0][j][i] = Rf_[0][i][j];
            Ru[0][j][i] = Ru_[0][i][j];
            Ro[0][j][i] = Ro_[0][i][j];
        }
    }
    

}

void split_4_tensor_B(float tensor_B[1][512], float Bi[1][128], float Bo[1][128], float Bf[1][128], float Bu[1][128], int STEPS) {
// Bias에서 Input Gate, Output Gate, Forget Gate, Update Gate에서 사용할 가중치를 분리하기 위한 함수  
    for (int i = 0; i < 128; i++) {
        if (i > STEPS) break;
        Bi[0][i] = tensor_B[0][i]; // Input Gate Bias 
        Bf[0][i] = tensor_B[0][STEPS + i]; // Output Gate Bias 
        Bu[0][i] = tensor_B[0][2 * STEPS + i] ; // Forget Gate Bias 
        Bo[0][i] = tensor_B[0][3 * STEPS + i] ; // Update Gate Bias 
    }
}

#endif