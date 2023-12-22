# 디렉토리 경로
directory_path = "../Weight/"

# 파일 이름들
file_names = [
    "tensor_W_1_16.txt", "tensor_R_1_16.txt", "tensor_B_1_16.txt",
    "tensor_W_2_16.txt", "tensor_R_2_16.txt", "tensor_B_2_16.txt",
    "tensor_W_3_16.txt", "tensor_R_3_16.txt", "tensor_B_3_16.txt",
    "tensor_W_4_1.txt",  "tensor_R_4_1.txt",  "tensor_B_4_1.txt"
]

# 파일 읽기 함수
def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def process_tensor_b(data, num):
    elements = [float(x) for x in data.split()]
    processed_elements = [(elements[i] + elements[i + num]) for i in range(num)]
    return '{' + ', '.join(map(str, processed_elements)) + '}'

# 1차원 C 형식 배열로 변환하는 함수
def convert_to_c_array_1d(data):
    elements = data.split()
    return '{' + ', '.join(elements) + '}'

# 2차원 C 형식 배열로 변환하는 함수 (특별한 처리: tensor_R_*)
def convert_to_c_array_2d(data, num):
    elements = data.split()
    array_2d = '{'
    for i in range(0, len(elements), num):
        row = '{' + ', '.join(elements[i:i + num]) + '}'
        array_2d += row + ', '
    array_2d = array_2d.rstrip(', ') + '}'
    return array_2d

# 파일에 데이터 쓰기 함수
def write_file(file_path, data):
    with open(file_path, 'w') as file:
        file.write(data)

# 각 파일에 대해 처리
for file_name in file_names:
    full_path = directory_path + file_name
    data = read_file(full_path)
    if 'tensor_R_' in file_name:
        # 2차원 배열로 변환, 'num' 값은 가정에 따라 설정
        if 'tensor_R_4_1.txt' in file_name:
            num = 1  # 예를 들어 num을 4로 설정
            c_array = convert_to_c_array_2d(data, num)
        else:
            num = 16  # 예를 들어 num을 4로 설정
            c_array = convert_to_c_array_2d(data, num)
    elif 'tensor_B_' in file_name:
        if 'tensor_B_4_1.txt' in file_name: 
            num = 4
            c_array = process_tensor_b(data, num)
        else:
            num = 64  # 예를 들어 num을 4로 설정
        c_array = process_tensor_b(data, num)
    else:
        # 기본 1차원 배열로 변환
        c_array = convert_to_c_array_1d(data)

    output_file_name = file_name.replace('tensor', 'converted_tensor')
    output_full_path = directory_path + output_file_name
    write_file(output_full_path, c_array)





