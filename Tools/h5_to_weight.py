from keras.models import load_model
import numpy as np
import os

dir_path = './Model/'

file_name = 'DEH_epoch1_128_model0.h5'

# 모델 불러오기
model = load_model(dir_path + file_name)

# 저장할 디렉토리 지정
save_dir = '../Weight'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 모델의 레이어 이름을 리스트로 저장
# 이미지에 표시된 레이어 이름에 따라 이 부분을 수정하세요.
layer_names = ['lstm_4', 'lstm_5', 'lstm_6', 'lstm_7']

# 각 레이어의 가중치를 추출하고 .npy 파일로 저장
for layer_index, layer_name in enumerate(layer_names, start=1):
    # 가중치 추출
    weights = model.get_layer(layer_name).get_weights()
    weight_types = ['W', 'R', 'B']
    for i, weight in enumerate(weights):
        weight_type = weight_types[i]
        if weight_type == 'B':
            weight_dimension = weight.shape[0]//4
        elif weight_type == 'W':
            weight_dimension = weight.shape[1]//4
        elif weight_type =='R':
            weight_dimension = weight.shape[1]//4
        # 파일명 형식: "Weight_test/layername_weighttype.npy"
        weight_path = os.path.join(save_dir, f'tensor_{weight_type}_{layer_index}_{weight_dimension}.npy')
        np.save(weight_path, weight)