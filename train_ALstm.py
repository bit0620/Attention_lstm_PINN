from utiles.Attention_lstm_model import Attention_lstm_model
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR

from utiles.function import *
from torch.optim import Adam
from torch import nn
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def train(optiomizer, num_epochs, net, train_iter, params, lambda_weight):
    net = net.cpu()
    Optim = optiomizer
    num_epochs = num_epochs
    schedular = MultiStepLR(Optim, milestones=[50, 100], gamma=0.1)
    loss_var = []
    for epoch in tqdm(range(num_epochs)):
        train_loss = []
        for (X, Y), param in zip(train_iter, params):
            X, Y = X.cpu(), Y.cpu()
            param = param.cpu()
            Optim.zero_grad()
            Y_hat = net(X)
            #l = nn.MSELoss()(Y_hat, Y)
            l = loss_function(Y_hat, Y, param, lambda_weight)
            l.backward()
            Optim.step()
            train_loss.append(l.detach().clone())

        train_loss = torch.tensor(train_loss)
        schedular.step()

        epoch_loss = torch.mean(train_loss).squeeze()
        loss_var.append(epoch_loss)

    loss_var = torch.tensor(loss_var)
    loss_var = loss_var.numpy()

    # plt.figure()
    # plt.plot(np.arange(num_epochs), loss_var)
    # plt.title("Training Curve")
    # plt.show()

    torch.save(net, r'model\ALstm_.pt')


def test(tset_iter, loss_fn):
    net = torch.load(r'model\CALstm.pt', weights_only=False, map_location='cpu')


    # net = net.cpu()
    net = net.cpu()

    loss = loss_fn

    Y_pred = []
    Y_real = []

    test_loss = []
    test_map = []
    test_mape = []
    test_property_corr = []

    with torch.no_grad():
        for X, Y in tset_iter:
            # X, Y = X.cpu(), Y.cpu()
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


        property_corr = torch.mean(torch.cat(test_property_corr, dim=0)).squeeze().float()
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

        plot_len = 120

        plt.plot(np.arange(plot_len), pred_price[0:plot_len], color='red', label='predict price')
        plt.plot(np.arange(plot_len), true_price[0:plot_len], color='blue', label='true price')
        plt.legend()
        plt.title("真实价格与预测价格")
        plt.xlabel('时间')
        plt.ylabel('期权价格')
        plt.show()

        return torch.tensor(result)

if __name__ == '__main__':
    batch_size, num_epoches, learn_rate, num_workers = 128, 150, 0.001, 4
    in_channel, input_size, hd_size, num_lstm_layers, drop_out, num_attention_heads = 3, 5, 32, 2, 0.3, 4
    out_channel, middle_channel, num_steps, lambda_weight = 3, 6, 10, 0.02


    params = [out_channel, middle_channel, hd_size, learn_rate, num_lstm_layers, lambda_weight]

    train_data = torch.load(r'data\input_train.pt').float()
    train_data_mean = torch.mean(train_data, dim=0, keepdim=True).float()
    train_data_std = torch.std(train_data, dim=0, keepdim=True).float()
    train_data_normalization = Normalization(train_data_mean, train_data_std)
    train_input = train_data_normalization.normalize(train_data)

    train_target = torch.load(r'data\label_train.pt').float()
    train_target_mean = torch.mean(train_target, dim=0, keepdim=True).float()
    train_target_std = torch.std(train_target, dim=0, keepdim=True).float()
    train_target_normalization = Normalization(train_target_mean, train_target_std)
    train_target = train_target_normalization.normalize(train_target)

    train_iter = DataLoader(TensorDataset(train_input, train_target), batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    test_data = torch.load(r'data\input_test.pt').float()
    test_data_mean = torch.mean(test_data, dim=0, keepdim=True).float()
    test_data_std = torch.std(test_data, dim=0, keepdim=True).float()
    test_data_normalization = Normalization(test_data_mean, test_data_std)
    test_input = train_data_normalization.normalize(test_data)

    test_target = torch.load(r'data\label_test.pt').float()
    test_target_mean = torch.mean(test_target, dim=0, keepdim=True).float()
    test_target_std = torch.std(test_target, dim=0, keepdim=True).float()
    test_target_normalization = Normalization(test_target_mean, test_target_std)
    test_target = train_target_normalization.normalize(test_target)

    loss_data = torch.load(r'data/loss_data_process.pt').float()

    params_loss = DataLoader(loss_data, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    test_iter = DataLoader(TensorDataset(test_input, test_target), batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    model = Attention_lstm_model(in_channel=in_channel, input_size=input_size, lstm_hdsize=hd_size, num_lstm_layers=num_lstm_layers, drop_out=drop_out, num_heads=num_attention_heads,
                                 out_channel=out_channel, middle_channel=middle_channel, num_step=num_steps)

    optimizer = Adam(model.parameters(), lr=learn_rate)
    loss_test_fn = nn.MSELoss()

    train(optimizer, num_epoches, model, train_iter, params_loss, lambda_weight)
    result = test(test_iter, loss_test_fn)

    # params = np.array(params)
    # result = np.array(torch.Tensor.cpu(result))
    #
    # params_result = np.concatenate([params, result], axis=0)
    # params_result = params_result.reshape((1, -1))
    # df = pd.DataFrame(params_result)
    # df.to_csv(r'data\params_result.csv', index=False, mode='a', header=False)

