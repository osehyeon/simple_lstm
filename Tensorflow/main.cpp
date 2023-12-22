#include <iostream>
#include "LSTMOps.cpp"
#include "WeightSplit.cpp"
#include "WeightResize.cpp"
//  128 -> LSTM32 -> LSTM64 -> LSTM128 -> LSTM32 -> LSTM64 -> LSTM1

Tensor LSTM(Tensor tensor_X, float tensor_W[1][512][1],  float tensor_R[1][128][512], float tensor_B[1][512], int I_STEPS, int O_STEPS);

int tensor_X_resize(const char *file_path, Tensor *tensor);

int main() 

{
    // 가중치 파일의 경로를 가져와 저장함
    char file_W_1[] = "../Weight/tensor_W_1.txt";
    char file_R_1[] = "../Weight/tensor_R_1.txt";
    char file_B_1[] = "../Weight/tensor_B_1.txt";
    char file_W_2[] = "../Weight/tensor_W_2.txt";
    char file_R_2[] = "../Weight/tensor_R_2.txt";
    char file_B_2[] = "../Weight/tensor_B_2.txt";

    char file_W_3[] = "../Weight/tensor_W_3.txt";
    char file_B_3[] = "../Weight/tensor_B_3.txt";
    
    char file_W_4[] = "../Weight/tensor_W_4.txt";
    char file_B_4[] = "../Weight/tensor_B_4.txt";

    char file_W_5[] = "../Weight/tensor_W_5.txt";
    char file_B_5[] = "../Weight/tensor_B_5.txt";


    // 가중치를 저장할 변수 생성 
    float tensor_W_1[1][512][1] = {{{0}}};
    float tensor_R_1[1][128][512]= {{{0}}}; 
    float tensor_B_1[1][512]= {{{0}}};

    float tensor_W_2[1][512][1] = {{{0}}};
    float tensor_R_2[1][128][512]= {{{0}}}; 
    float tensor_B_2[1][512]= {{{0}}};

    float tensor_W_3[1][128][128] = {{{0}}};
    float tensor_B_3[1][128] = {{{0}}}; 

    float tensor_W_4[1][128][128] = {{{0}}};
    float tensor_B_4[1][128] = {{{0}}}; 

    float tensor_W_5[1][128][128] = {{{0}}};
    float tensor_B_5[1][128] = {{{0}}}; 


    // 가중치 파일 내 데이터를 가중치 변수에 저장 
    tensor_W_resize(file_W_1, tensor_W_1, 64);
    tensor_R_resize(file_R_1, tensor_R_1, 64);
    tensor_B_resize(file_B_1, tensor_B_1, 64);

    tensor_W_resize(file_W_2, tensor_W_2, 128);
    tensor_R_resize(file_R_2, tensor_R_2, 128);
    tensor_B_resize(file_B_2, tensor_B_2, 128);

    tensor_DW_resize(file_W_3, tensor_W_3, 128, 64);
    tensor_DB_resize(file_B_3, tensor_B_3, 64);    

    tensor_DW_resize(file_W_4, tensor_W_4, 64, 128);
    tensor_DB_resize(file_B_4, tensor_B_4, 128);    
    
    tensor_DW_resize(file_W_5, tensor_W_5, 128, 1);
    tensor_DB_resize(file_B_5, tensor_B_5, 1);    

    // 비정상 데이터 파일의 경로를 바탕으로 입력 데이터 배열에 데이터를 저장 
    char tensor_X_128[] = "../Data/anormal.txt"; // ../data/normal.txt 파일을 지정 시 정상 데이터일 때 추론 결과를 확인할 수 있음 
    Tensor tensor_X = {{{0}}};
    tensor_X_resize(tensor_X_128, &tensor_X);
    
    // LSTM 연산 진행 (16, 16, 16, 1)
    tensor_X = LSTM(tensor_X, tensor_W_1, tensor_R_1, tensor_B_1, 128, 64);
    tensor_X = LSTM(tensor_X, tensor_W_2, tensor_R_2, tensor_B_2, 64, 128);

    tensor_X = Dense(tensor_X, tensor_W_3, tensor_B_3, 128, 64);
    tensor_X = compute_relu(tensor_X, 64);

    tensor_X = Dense(tensor_X, tensor_W_4, tensor_B_4, 64, 128);
    tensor_X = compute_relu(tensor_X, 128);
    
    tensor_X = Dense(tensor_X, tensor_W_5, tensor_B_5, 128, 1);
    tensor_X = compute_sigmoid(tensor_X, 1);

    for(int i=0; i<1; i++) {
        printf("%f\n", tensor_X.data[0][0][i]);
    }

    return 0;
}

Tensor LSTM(Tensor tensor_X, float tensor_W[1][512][1],  float tensor_R[1][128][512], float tensor_B[1][512], int I_STEPS, int O_STEPS) 
// LSTM Ops 파일의 핵심 연산 모듈을 바탕으로 LSTM 연산을 구현하는 코드 
{
    // 내부 가중치 정의 (weight, Recurrence Weight, Bias) 中 (Input, Output, Forget, Update) 
    float Wi[1][128][1] = {{{0}}};
    float Wo[1][128][1] = {{{0}}};
    float Wf[1][128][1] = {{{0}}};
    float Wu[1][128][1] = {{{0}}};
    float Ri[1][128][128] = {{{0}}};
    float Ro[1][128][128] = {{{0}}};
    float Rf[1][128][128] = {{{0}}};
    float Ru[1][128][128] = {{{0}}};
    float Bi[1][128] = {{0}};
    float Bo[1][128] = {{0}};
    float Bf[1][128]= {{0}};
    float Bu[1][128] = {{0}};

    // 전체 가중치에서 Input, Output, Forget, Update 가중치를 분리하여 배열에 저장 
    split_4_tensor_W(tensor_W, Wi, Wo, Wf, Wu, O_STEPS);
    split_4_tensor_R(tensor_R, Ri, Ro, Rf, Ru, O_STEPS);
    split_4_tensor_B(tensor_B, Bi, Bo, Bf, Bu, O_STEPS);
    
    // 입출력을 받아올 변수 정의 
    Tensor Y_h = {{{0}}};
    Tensor Y_c = {{{0}}};
    Tensor Y_c_temp = {{{0}}};
    Tensor it = {{{0}}};
    Tensor ot = {{{0}}};
    Tensor ft = {{{0}}};
    Tensor ct = {{{0}}};

    // 입력 데이터 변수 정의 
    float X[1][1][1]= {{{0}}};

    // LSTM 연산 진행 
    for(int s=0; s<128; s++) {
        if(s >= I_STEPS) {
            break;
        }
        X[0][0][0] = tensor_X.data[0][0][s];
        it = compute_gate(X, Wi, Ri, Bi, Y_h, O_STEPS);
        it = compute_sigmoid(it, O_STEPS);

        ot = compute_gate(X, Wo, Ro, Bo, Y_h, O_STEPS);
        ot = compute_sigmoid(ot, O_STEPS);

        ft = compute_gate(X, Wf, Rf, Bf, Y_h, O_STEPS);
        ft = compute_sigmoid(ft, O_STEPS);

        ct = compute_gate(X, Wu, Ru, Bu, Y_h, O_STEPS);
        ct = compute_tanh(ct, O_STEPS);

        Y_c = compute_cell_state(ft, it, ct, Y_c, O_STEPS);
        Y_c_temp = compute_tanh(Y_c, O_STEPS);
        Y_h = compute_hidden_state(ot, Y_c_temp, Y_h, O_STEPS);

    }  
    return Y_h;
}

int tensor_X_resize(const char *file_path, Tensor *tensor) {
// 텍스트 형식의 입력 데이터(정상, 비정상)를 불러와 배열에 저장하기 위해 사용  
    FILE *file;
    int count = 0;

    file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1; 
    }

    while (fscanf(file, "%f", &tensor->data[0][0][count]) == 1) {
        count++;
        if (count >= 128) {  
            break;
        }
    }

    fclose(file);

    return 0;  
}
