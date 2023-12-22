#include <iostream>
#include <string.h>
#include "LSTMOps.cpp"
#include "WeightSplit.cpp"
#include "WeightResize.cpp"
//  128 -> LSTM32 -> LSTM64 -> LSTM128 -> LSTM32 -> LSTM64 -> LSTM1

Tensor LSTM(Tensor tensor_X, float tensor_W[1][512][1],  float tensor_R[1][512][128], float tensor_B[1][1024], int I_STEPS, int O_STEPS);
int tensor_X_resize(const char *file_path, Tensor *tensor);
void create_file_path(char *dir, char *file_name, char *full_path);
int main() 
// LSTM(16,16,16,1) 모델 생성 및 실행 
{
    char dir_path[] = "../Weight/";

    // Define file names
    char file_name_W_1[] = "tensor_W_1_16.txt";
    char file_name_R_1[] = "tensor_R_1_16.txt";
    char file_name_B_1[] = "tensor_B_1_16.txt";

    char file_name_W_2[] = "tensor_W_2_16.txt";
    char file_name_R_2[] = "tensor_R_2_16.txt";
    char file_name_B_2[] = "tensor_B_2_16.txt";

    char file_name_W_3[] = "tensor_W_3_16.txt";
    char file_name_R_3[] = "tensor_R_3_16.txt";
    char file_name_B_3[] = "tensor_B_3_16.txt";

    char file_name_W_4[] = "tensor_W_4_1.txt";
    char file_name_R_4[] = "tensor_R_4_1.txt";
    char file_name_B_4[] = "tensor_B_4_1.txt";

    char file_W_1[100] = "";
    char file_R_1[100] = "";
    char file_B_1[100] = "";

    char file_W_2[100] = "";
    char file_R_2[100] = "";
    char file_B_2[100] = "";

    char file_W_3[100] = "";
    char file_R_3[100] = "";
    char file_B_3[100] = "";

    char file_W_4[100] = "";
    char file_R_4[100] = "";
    char file_B_4[100] = "";

    create_file_path(dir_path, file_name_W_1, file_W_1);
    create_file_path(dir_path, file_name_R_1, file_R_1);
    create_file_path(dir_path, file_name_B_1, file_B_1);

    create_file_path(dir_path, file_name_W_2, file_W_2);
    create_file_path(dir_path, file_name_R_2, file_R_2);
    create_file_path(dir_path, file_name_B_2, file_B_2);

    create_file_path(dir_path, file_name_W_3, file_W_3);
    create_file_path(dir_path, file_name_R_3, file_R_3);
    create_file_path(dir_path, file_name_B_3, file_B_3);

    create_file_path(dir_path, file_name_W_4, file_W_4);
    create_file_path(dir_path, file_name_R_4, file_R_4);
    create_file_path(dir_path, file_name_B_4, file_B_4);


    printf("%s\n", file_W_1);
    // 가중치를 저장할 변수 생성 
    float tensor_W_1_16[1][512][1] = {{{0}}};
    float tensor_R_1_16[1][512][128]= {{{0}}}; 
    float tensor_B_1_16[1][1024]= {{{0}}};

    float tensor_W_2_16[1][512][1] = {{{0}}};
    float tensor_R_2_16[1][512][128] = {{{0}}};
    float tensor_B_2_16[1][1024] = {{{0}}};

    float tensor_W_3_16[1][512][1] = {{{0}}};
    float tensor_R_3_16[1][512][128] = {{{0}}};
    float tensor_B_3_16[1][1024] = {{{0}}};

    float tensor_W_4_1[1][512][1] = {{{0}}};
    float tensor_R_4_1[1][512][128] = {{{0}}};
    float tensor_B_4_1[1][1024] = {{{0}}};

    // 가중치 파일 내 데이터를 가중치 변수에 저장 
    tensor_W_resize(file_W_1, tensor_W_1_16, 16);
    tensor_R_resize(file_R_1, tensor_R_1_16, 16);
    tensor_B_resize(file_B_1, tensor_B_1_16, 16);
    
    tensor_W_resize(file_W_2, tensor_W_2_16, 16);
    tensor_R_resize(file_R_2, tensor_R_2_16, 16);
    tensor_B_resize(file_B_2, tensor_B_2_16, 16);

    tensor_W_resize(file_W_3, tensor_W_3_16, 16);
    tensor_R_resize(file_R_3, tensor_R_3_16, 16);
    tensor_B_resize(file_B_3, tensor_B_3_16, 16);

    tensor_W_resize(file_W_4, tensor_W_4_1, 1);
    tensor_R_resize(file_R_4, tensor_R_4_1, 1);
    tensor_B_resize(file_B_4, tensor_B_4_1, 1);

    // 비정상 데이터 파일의 경로를 바탕으로 입력 데이터 배열에 데이터를 저장 
    char tensor_X_128[] = "../Data/anormal.txt"; // ../data/normal.txt 파일을 지정 시 정상 데이터일 때 추론 결과를 확인할 수 있음 
    Tensor tensor_X = {{{0}}};
    tensor_X_resize(tensor_X_128, &tensor_X);
    
    // LSTM 연산 진행 (16, 16, 16, 1)
    tensor_X = LSTM(tensor_X, tensor_W_1_16, tensor_R_1_16, tensor_B_1_16, 128, 16);
    tensor_X = LSTM(tensor_X, tensor_W_2_16, tensor_R_2_16, tensor_B_2_16, 16, 16);
    tensor_X = LSTM(tensor_X, tensor_W_3_16, tensor_R_3_16, tensor_B_3_16, 16, 16);
    tensor_X = LSTM(tensor_X, tensor_W_4_1, tensor_R_4_1, tensor_B_4_1, 16, 1);
    //tensor_X = compute_sigmoid(tensor_X, 1);

    // 추론 결과 출력 (0에 가까우면 정상, 1에 가까우면 비정상)
    for(int i=0; i<1; i++)
    printf("%f\n", tensor_X.data[0][0][i]);
    
    return 0;
}


Tensor LSTM(Tensor tensor_X, float tensor_W[1][512][1],  float tensor_R[1][512][128], float tensor_B[1][1024], int I_STEPS, int O_STEPS) 
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

void create_file_path(char *dir, char *file_name, char *full_path) {
    strcpy(full_path, dir);       // Copy directory path to full path
    strcat(full_path, file_name); // Append file name to full path
}
