#include <stdio.h>
#include <stdlib.h>

void tensor_W_resize(const char *file_path, float array[1][512][1], int STEPS) {
// Weights 바이너리 파일을 읽어와 배열에 저장하는 코드 
    FILE *file;
    file = fopen(file_path, "rb"); 
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    fread(array, sizeof(float), 4*STEPS, file);

    fclose(file);
}

void tensor_R_resize(const char *file_path, float tensor_W[1][512][128], int STEPS) {
// Recurrence Weights 바이너리 파일을 읽어와 배열에 저장하는 코드 
    FILE *file;
    file = fopen(file_path, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    for (int i = 0; i < 4 * STEPS; ++i) {
        for (int j = 0; j < STEPS; ++j) {
            if (fread(&tensor_W[0][i][j], sizeof(float), 1, file) != 1) {
                break;
            }
        }
    }
    fclose(file);
}


void tensor_B_resize(const char *file_path, float array[1][1024], int STEPS) {
// Bias 바이너리 파일을 읽어와 배열에 저장하는 코드 
    FILE *file;
    file = fopen(file_path, "rb");  
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    fread(array, sizeof(float), 4*2*STEPS, file);

    fclose(file);
}
