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

void tensor_R_resize(const char *file_path, float array[1][128][512], int STEPS) {
// Recurrence Weights 텍스트 파일을 읽어와 배열에 저장하는 코드 
    FILE *file;
    int count = 0;

    file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Error opening file");
    }

    while (fscanf(file, "%f", &array[0][count / (STEPS * 4)][count % (STEPS * 4)]) == 1) {
        count++;
        if (count >= 4 * STEPS * STEPS) { 
            break;
        }
    }
    fclose(file);
}

void tensor_B_resize(const char *file_path, float array[1][512], int STEPS) {
// Bias 텍스트 파일을 읽어와 배열에 저장하는 코드 
    FILE *file;
    int count = 0;

    file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Error opening file");
    }

    while (fscanf(file, "%f", &array[0][count]) == 1) {
        count++;
        if (count >= 4 * STEPS) {
            break;
        }
    }

    fclose(file);

}

void tensor_DW_resize(const char *file_path, float array[1][128][128], int I_STEPS, int O_STEPS) {
// Recurrence Weights 텍스트 파일을 읽어와 배열에 저장하는 코드 
    FILE *file;
    int count = 0;

    file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Error opening file");
    }

    while (fscanf(file, "%f", &array[0][count / (O_STEPS)][count % (O_STEPS)]) == 1) {
        count++;
        if (count >= I_STEPS * O_STEPS) { 
            break;
        }
    }
    fclose(file);
}

void tensor_DB_resize(const char *file_path, float array[1][128], int STEPS) {
// Bias 텍스트 파일을 읽어와 배열에 저장하는 코드 
    FILE *file;
    int count = 0;

    file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Error opening file");
    }

    while (fscanf(file, "%f", &array[0][count]) == 1) {
        count++;
        if (count >= STEPS) {
            break;
        }
    }

    fclose(file);

}

