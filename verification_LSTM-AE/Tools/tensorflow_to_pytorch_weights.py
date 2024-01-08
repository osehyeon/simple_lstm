import numpy as np
import os 

def load_weights_from_file(file_path, Weight, H):
    data = np.load(file_path)
    if Weight == 'W':
        input_gate = data[0][0:H]
        forget_gate = data[0][H:2*H]
        update_gate = data[0][2*H:3*H]
        output_gate = data[0][3*H:4*H]
        result = np.concatenate([input_gate, output_gate, forget_gate, update_gate])
      
    elif Weight == 'R':
        reshaped_data = data.reshape(H, 4*H)
        input_gates = reshaped_data[:, 0:H]
        forget_gates = reshaped_data[:, H:2*H]
        update_gates = reshaped_data[:, 2*H:3*H]
        output_gates = reshaped_data[:, 3*H:4*H]
        
        r_input_gates = np.transpose(input_gates)
        r_forget_gates = np.transpose(forget_gates)
        r_update_gates = np.transpose(update_gates)
        r_output_gates = np.transpose(output_gates)
        
        result = np.concatenate([r_input_gates, r_output_gates, r_forget_gates, r_update_gates])
        
    elif Weight == 'B':
        input_gate = data[0:H]
        #print(input_gate)
        forget_gate = data[H:2*H]
        #print(forget_gate)
        update_gate = data[2*H:3*H]
        #print(update_gate)
        output_gate = data[3*H:4*H]
        #print(output_gate)
        zero_gate = [0] * 4*H
        
        result = np.concatenate([input_gate, output_gate, forget_gate, update_gate, zero_gate])
    
    return result

def extract_info_from_filename(file_path):
    parts = file_path.split('_')  # 파일 이름을 '_'를 기준으로 분할
    Weight = parts[1]             # 두 번째 부분 (예: 'W')
    H = int(parts[3].split('.')[0])  # 세 번째 부분에서 숫자 추출 (예: '1.npy'에서 '1'을 추출하고 정수로 변환)

    return Weight, H


directory = '../Weight'

files = [
    "tensor_W_1_128.npy", 
    "tensor_R_1_128.npy", 
    "tensor_B_1_128.npy"
]


for file in files:
    # 파일 경로 구성
   
    file_path = os.path.join(directory, file)
    
    Weight, H = extract_info_from_filename(file)
    #print(Weight, H)
    array = load_weights_from_file(file_path, Weight, H)
    
    
    # txt 파일 경로 설정 (.npy 확장자를 .txt로 변경)
    txt_path = file_path.replace('.npy', '.txt')
    
    # 텍스트 파일로 저장
    with open(txt_path, 'w') as txt_file:
        for item in array.flatten():
            txt_file.write(f"{item}\n")
    print(f"Saved {file} to {txt_path}")

    