import numpy as np
import matplotlib.pyplot as plt

# 텍스트 파일로부터 데이터를 로드하는 함수
def load_data_from_txt(filename):
    with open(filename, 'r') as file:
        data = [float(line.strip()) for line in file]
    return np.array(data)

# 정상 데이터와 비정상 데이터 파일 경로
normal_data_path = '../Data/normalization/normal.txt' 
abnormal_data_path = '../Data/normalization/anormal.txt'

# 데이터 로드
normal_data_loaded = load_data_from_txt(normal_data_path)
abnormal_data_loaded = load_data_from_txt(abnormal_data_path)

# 로드된 데이터의 시각화
plt.figure(figsize=(14, 6))

# 정상 데이터 시각화
plt.subplot(1, 2, 1)
plt.title("Visualizing Normal Data")
plt.plot(normal_data_loaded, label='Normal Data')
plt.legend()

# 비정상 데이터 시각화
plt.subplot(1, 2, 2)
plt.title("Visualizing Abnormal Data")
plt.plot(abnormal_data_loaded, label='Abnormal Data', color='red')
plt.legend()

plt.show()
