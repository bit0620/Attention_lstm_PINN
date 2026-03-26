from tqdm import trange
import pandas as pd
import numpy as np
import torch
import os

seq_len = 10

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
parent_dir = os.path.dirname(current_dir)
# 构建正确的数据目录路径
file_path = os.path.join(parent_dir, 'data')
csv_path_train = os.path.join(file_path, 'train_data.csv')
csv_path_test = os.path.join(file_path, 'test_data.csv')


for j in range(2):
    csv_path = (csv_path_train if j == 0 else csv_path_test)
    data = pd.read_csv(csv_path)
    data_input = []
    data_label = []

    for i in trange(data['optID'].unique().shape[0]):
        for index in range(data.loc[(data['optID'] == data['optID'].unique()[i])].shape[0] - seq_len + 1):
            input = torch.zeros((3, 10, 5))
            tmp_input = np.array(data.loc[(data['optID'] == data['optID'].unique()[i])][index:index + seq_len])
            input[0, :] = torch.tensor(np.array(tmp_input[0:seq_len, 3:8], dtype=np.float64))
            input[1, :] = torch.tensor(np.array(tmp_input[0:seq_len, 8:13], dtype=np.float64))
            input[2, :] = torch.tensor(np.array(tmp_input[0:seq_len, 13:18], dtype=np.float64))
            label = torch.tensor(np.array(tmp_input[seq_len - 1, 18], dtype=np.float64)).cpu()
            data_input.append(input)
            data_label.append(label)

    data_label = torch.tensor(data_label)
    zeros = torch.zeros((len(data_input), 3, 10, 5))
    for i in range(zeros.shape[0]):
        zeros[i] = data_input[i]
    data_input = zeros

    print()
#     if csv_path == csv_path_train:
#         torch.save(data_input, r'..\data\\input_train.pt')
#         torch.save(data_label, r'..\data\label_train.pt')
#     else:
#         torch.save(data_input, r'..\data\input_test.pt')
#         torch.save(data_label, r'..\data\label_test.pt')


# loss_data = pd.read_csv(r'..\data\loss_data.csv')
# data_input_loss = []
#
# for i in trange(loss_data['optID'].unique().shape[0]):
#     for index in range(loss_data.loc[(loss_data['optID'] == loss_data['optID'].unique()[i])].shape[0] - seq_len + 1):
#         input_loss = torch.zeros((6))
#         tmp_input = np.array(loss_data.loc[(loss_data['optID'] == loss_data['optID'].unique()[i])][index:index + seq_len])
#         input_loss = torch.tensor(np.array(tmp_input[seq_len - 1, 1:], dtype=np.float64))
#         data_input_loss.append(input_loss)
#
# zeros = torch.zeros((len(data_input_loss), 6))
# for i in range(zeros.shape[0]):
#     zeros[i] = data_input_loss[i]
# data_input_loss = zeros
#
# torch.save(data_input_loss, r'..\data\loss_data_process.pt')

