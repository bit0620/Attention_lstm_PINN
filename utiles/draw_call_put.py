import sys
import os
import torch
# --- 路径设置开始 ---
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前目录的上一级目录（即项目根目录 Attention_lstm_PINN）
parent_dir = os.path.dirname(current_dir)
# 将项目根目录添加到 Python 的搜索路径中
sys.path.append(parent_dir)
# --- 路径设置结束 ---
model = torch.load(os.path.join(parent_dir, "model", "ALstm_9_pi"), 
                   weights_only=False, 
                   map_location=torch.device('cpu'))




from utiles.Attention_lstm_model import Attention_lstm_model
from torch.utils.data import DataLoader, TensorDataset
from utiles.function import *
from tqdm import trange

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_data(seq_len=10):
    data = pd.read_csv("data/test_data.csv")

    call_data = data[data.iloc[:, 4] == 1]
    put_data = data[data.iloc[:, 4] == 0]

    call_moneyness = call_data["spotPrice"] / call_data["strikePrice"]
    call_OTM = call_data[call_moneyness < 0.98]  # 虚值（OTM）
    call_ATM = call_data[(call_moneyness >= 0.98) & (call_moneyness <= 1.02)]  # 平值（ATM）
    call_ITM = call_data[call_moneyness > 1.02]

    put_moneyness = put_data["spotPrice"] / put_data["strikePrice"]
    put_DITM = put_data[put_moneyness < 0.95]
    put_OTM = put_data[(put_moneyness > 0.95) & (put_moneyness < 0.98)]
    put_ATM = put_data[(put_moneyness >= 0.98) & (put_moneyness <= 1.02)]
    put_ITM = put_data[(put_moneyness > 1.02) & (put_moneyness < 1.05)]
    put_DOTM = put_data[put_moneyness > 1.05]

    data_list = [call_OTM, call_ATM, call_ITM, put_OTM, put_ATM, put_ITM]

    # call = [call_data, put_data]
    # call = [data]
    data_input_result = []
    data_label_result = []

    for data in data_list:
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

        data_input_result.append(data_input)
        data_label_result.append(data_label)

    return data_input_result, data_label_result

def draw_ATM_OTM_ITM(true_prices, pred_prices):
    # 定义期权类型标签
    option_types_3 = ['OTM', 'ATM', 'ITM']
    option_types_2 = ["Call", "Put"]

    # 遍历每个子图绘制数据
    for row in range(2):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        # plt.tight_layout()
        for col in range(3):
            ax = axes[col]
            idx = row * 3 + col

            # 获取前120个数据点
            true = true_prices[idx][:120]
            pred = pred_prices[idx][:120]

            if row == 0 and col == 2:
                true = true_prices[idx][120:240]
                pred = pred_prices[idx][120:240]


            # 绘制折线图
            ax.plot(true, 'b-', label='true_price', linewidth=1.5)
            ax.plot(pred, 'r--', label='pred_price', linewidth=1.5)

            # 添加标签和标题
            ax.set_xlabel('time', fontsize=10)
            ax.set_ylabel('option_price', fontsize=10)
            ax.set_title(f'{option_types_2[row]} - {option_types_3[col]}', fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.show()

def caculate_error(test_iter, model):
    net = model.cpu()

    loss = nn.MSELoss()

    Y_pred = []
    Y_real = []

    test_loss = []
    test_map = []
    test_mape = []
    test_property_corr = []

    with torch.no_grad():
        for X, Y in test_iter:
            # X, Y = X.cuda(), Y.cuda()
            Y_hat = net(X)
            Y = test_target_normalization.unnormalize(Y)
            Y_hat = test_target_normalization.unnormalize(Y_hat)
            l = loss(Y, Y_hat)

            Y_pred.append(Y_hat.detach())
            Y_real.append(Y.detach())

            corr, map, mape = metrics(Y_hat.detach(), Y.detach())

            test_loss.append(l.item())
            test_map.append(map)
            test_mape.append(mape)
            test_property_corr.append(corr)


        property_corr = torch.mean(torch.cat(test_property_corr, dim=0)).squeeze().float().cpu()
        map = torch.mean(torch.cat(test_map, dim=0)).squeeze().float().cpu()
        mape = torch.mean(torch.cat(test_mape, dim=0)).squeeze().float().cpu()
        loss = torch.mean(torch.tensor(test_loss)).float().cpu()
        rmse = torch.sqrt(loss).float()
        print("MSE: {:.5f}\nRMSE: {:0.5f}\nMAP: {:0.5f}\nMAPE: {:0.5f}\nCorrelation: {:0.5f}\n".format(loss, rmse, map, mape, property_corr))

        result = [loss, rmse, map, mape, property_corr]

        pred_price = torch.cat(Y_pred, dim=0)
        true_price = torch.cat(Y_real, dim=0)

        pred_price = pred_price.cpu()
        true_price = true_price.cpu()

        pred_price = pred_price.numpy()
        true_price = true_price.numpy()

        return pred_price, true_price

if __name__ == "__main__":
    model = torch.load(os.path.join(parent_dir, "model", "ALstm_9_pi"), 
                   weights_only=False, 
                   map_location=torch.device('cpu'))

    call_input_result, call_label_result = load_data()

    test_target = torch.load(os.path.join(parent_dir, "data", "label_test.pt")).float()

    test_target_mean = torch.mean(test_target, dim=0, keepdim=True).float()
    test_target_std = torch.std(test_target, dim=0, keepdim=True).float()
    test_target_normalization = Normalization(test_target_mean, test_target_std)

    true_prices_list = []
    pred_prices_list = []

    for input, label in zip(call_input_result, call_label_result):
        label_n = test_target_normalization.normalize(label)
        test_iter = DataLoader(TensorDataset(input, label_n), batch_size=256, num_workers=4, shuffle=False, drop_last=False)
        pred_price, true_price = caculate_error(test_iter, model)

        true_prices_list.append(true_price)
        pred_prices_list.append(pred_price)

    draw_ATM_OTM_ITM(true_prices_list, pred_prices_list)

