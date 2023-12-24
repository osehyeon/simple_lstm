#include <stdio.h>
#include <stdlib.h>

void tensor_W_resize(const char *file_path, float array[1][512][1], int STEPS) {
// Weights 텍스트 파일을 읽어와 배열에 저장하는 코드 
    FILE *file;
    int count = 0;

    file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Error opening file");
    }

    while (fscanf(file, "%f", &array[0][count][0]) == 1) {
        count++;
        if (count >= 4*STEPS) {
            break;
        }
    }

    fclose(file);

}

void tensor_R_resize(const char *file_path, float tensor_W[1][512][128], int STEPS) {
// Recurrence Weights 텍스트 파일을 읽어와 배열에 저장하는 코드 
    FILE *file;
    int count = 0;

    file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Error opening file");
    }

    while (fscanf(file, "%f", &tensor_W[0][count / STEPS][count % STEPS]) == 1) {
        count++;
        if (count >= 4 * STEPS * STEPS) { 
            break;
        }
    }

    fclose(file);

}

void tensor_B_resize(const char *file_path, float array[1][1024], int STEPS) {
// Bias 텍스트 파일을 읽어와 배열에 저장하는 코드 
    FILE *file;
    int count = 0;

    file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Error opening file");
    }

    while (fscanf(file, "%f", &array[0][count]) == 1) {
        count++;
        if (count >= 4 * 2 * STEPS) {
            break;
        }
    }

    fclose(file);

}


