import numpy as np
import os

# 지정된 디렉토리
directory = "./Weight_Npy/"

# 처리할 파일 목록
files = [
    "tensor_W_1_16.npy", "tensor_W_2_16.npy", "tensor_W_3_16.npy", "tensor_W_4_2.npy",
    "tensor_R_1_16.npy", "tensor_R_2_16.npy", "tensor_R_3_16.npy", "tensor_R_4_2.npy",
    "tensor_B_1_16.npy", "tensor_B_2_16.npy", "tensor_B_3_16.npy", "tensor_B_4_2.npy"
]

for file in files:
    # 파일 경로 구성
    file_path = os.path.join(directory, file)
    # NumPy 배열 로드
    array = np.load(file_path)
    # txt 파일 경로 설정 (.npy 확장자를 .txt로 변경)
    txt_path = file_path.replace('.npy', '.txt')
    
    # 텍스트 파일로 저장
    with open(txt_path, 'w') as txt_file:
        for item in array.flatten():
            txt_file.write(f"{item}\n")
    print(f"Saved {file} to {txt_path}")