from keras.models import load_model
import numpy as np
import os

file_name = 'lstm-ae_test.h5'

# 모델 불러오기
model = load_model(file_name)

# 저장할 디렉토리 지정
save_dir = '../Weight'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 모델의 레이어 이름을 리스트로 저장
layer_names = ['lstm']

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

layer_names = ['dense', 'dense_1', 'dense_2', 'dense_3', 'dense_4']

# 각 레이어의 가중치를 추출하고 .npy 파일로 저장
for layer_index, layer_name in enumerate(layer_names, start=2):
    # 가중치 추출
    weights = model.get_layer(layer_name).get_weights()
    weight_types = ['W', 'B']
    for i, weight in enumerate(weights):
        weight_type = weight_types[i]
        if weight_type == 'B':
            weight_dimension = weight.shape[0]
        elif weight_type == 'W':
            weight_dimension = weight.shape[1]
        weight_path = os.path.join(save_dir, f'tensor_{weight_type}_{layer_index}_{weight_dimension}.npy')
        np.save(weight_path, weight)