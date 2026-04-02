from torch.distributions.normal import Normal
from torch.autograd import grad
from scipy.stats import norm
from torch import nn

import numpy as np
import torch

class Normalization:
    def __init__(self, mean_val=None, std_val=None):
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x):
        return (x-self.mean_val)/self.std_val

    def unnormalize(self, x):
        return x*self.std_val + self.mean_val


def metrics(y, x):
    #x: reference signal
    #y: estimated signal
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    # corrlation
    x_mean = np.mean(x, axis=0, keepdims=True)
    y_mean = np.mean(y, axis=0, keepdims=True)
    x_std = np.std(x, axis=0, keepdims=True)
    y_std = np.std(y, axis=0, keepdims=True)
    corr = np.mean((x-x_mean)*(y-y_mean), axis=0, keepdims=True)/(x_std*y_std)

    # MAP
    map = np.mean(np.abs(x-y), axis=0, keepdims=True)

    # MAPE
    mape = np.mean(np.abs((x - y) / x), axis=0, keepdims=True)


    return torch.tensor(corr), torch.tensor(map), torch.tensor(mape)


# def black_scholes_price(type_, spot, strike, maturity, vol, r):
#     d1 = (torch.log(spot / strike) + (r + 0.5 * vol * vol) * maturity) / (vol * torch.sqrt(maturity))
#     d2 = d1 - vol * torch.sqrt(maturity)
#
#     type_ = type_.cpu().numpy()
#     d1_cdf = torch.tensor(norm.cdf(d1.cpu()), dtype=torch.float32).to(d1.device)
#     d2_cdf = torch.tensor(norm.cdf(d2.cpu()), dtype=torch.float32).to(d1.device)
#     neg_d1_cdf = torch.tensor(norm.cdf(-d1.cpu()), dtype=torch.float32).to(d1.device)
#     neg_d2_cdf = torch.tensor(norm.cdf(-d2.cpu()), dtype=torch.float32).to(d1.device)
#
#     call_prices = spot * d1_cdf - strike * torch.exp(-r * maturity) * d2_cdf
#     put_prices = -spot * neg_d1_cdf + strike * torch.exp(-r * maturity) * neg_d2_cdf
#
#     Y_bs = torch.where(torch.tensor(type_, dtype=torch.float32).to(spot.device) == 1, call_prices, put_prices)
#
#     return Y_bs

def black_scholes_price(type_, spot, strike, maturity, vol, r):
    d1 = torch.div(torch.log(torch.div(spot, strike)) + (r + 0.5 * torch.square(vol)) * maturity, vol * torch.sqrt(maturity))
    d2 = torch.div(torch.log(torch.div(spot, strike)) + (r - 0.5 * torch.square(vol)) * maturity, vol * torch.sqrt(maturity))
    device = spot.device
    dist = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
    cdf1, cdf2 = dist.cdf(d1), dist.cdf(d2)
    neg_1_cdf, neg_2_cdf = dist.cdf(-d1), dist.cdf(-d2)

    call = spot * cdf1 - strike * torch.exp(-r * maturity) * cdf2
    put = -spot * neg_1_cdf + strike * torch.exp(-r * maturity) * neg_2_cdf

    Y_bs = torch.where(type_ == 1, call, put)

    return Y_bs

def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True)[0]
        f = grads
        if grads is None:
            print('bad grad')
            return torch.tensor(0.)
    return grads

# Helper function for evaluating call option price from predicted volatility


# Helper function for PI-ConvTF Pinn Loss
def BS_PDE(params):
    type_ = params[:, 0]
    maturity = params[:, 1]
    vol = params[:, 2]
    strike = params[:, 3]
    spot = params[:, 4]
    r = params[:, 5]

    type_ = torch.unsqueeze(type_, dim=1)
    maturity = torch.unsqueeze(maturity, dim=1).requires_grad_(True)
    vol = torch.unsqueeze(vol, dim=1)
    strike = torch.unsqueeze(strike, dim=1)
    spot = torch.unsqueeze(spot, dim=1).requires_grad_(True)
    r = torch.unsqueeze(r, dim=1)

    c = black_scholes_price(type_, spot, strike, maturity, vol, r)

    c_t = nth_derivative(c, maturity, 1)
    c_s = nth_derivative(c, spot, 1)
    c_ss = nth_derivative(c, spot, 2)

    f = c_t + r*c - r*spot*c_s - torch.square(vol*spot)*c_ss*0.5
    # f = c_t + r * Y_hat - r * spot * c_s - torch.square(vol * spot) * c_ss * 0.5
    return f


def loss_function(Y_hat, Y, params, lambda_weight):
    # loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()

    loss_pred = loss_fn(Y_hat, Y)
    # loss_pred = nn.MSELoss()(Y_hat, Y)

    call_pde = BS_PDE(params)
    pi_loss_target = torch.zeros_like(call_pde)
    loss_pi = loss_fn(call_pde, pi_loss_target)

    loss_pred += loss_pi * lambda_weight

    return loss_pred

# def black_scholes_derivatives(type_, spot, strike, maturity, vol, r):
#     d1 = (torch.log(spot / strike) + (r + 0.5 * vol * vol) * maturity) / (vol * torch.sqrt(maturity))
#     d2 = d1 - vol * torch.sqrt(maturity)
#
#     # Ensure that norm.cdf works with tensors by converting the result to tensor
#     d1_cdf = torch.tensor(norm.cdf(d1.cpu()), dtype=torch.float32).to(d1.device)
#     d2_cdf = torch.tensor(norm.cdf(d2.cpu()), dtype=torch.float32).to(d1.device)
#     d1_pdf = torch.tensor(norm.pdf(d1.cpu()), dtype=torch.float32).to(d1.device)
#
#     # delta_t = (-spot * d1_cdf * (r + 0.5 * vol * vol) / (vol * torch.sqrt(maturity)) -
#     #                   strike * torch.exp(-r * maturity) * d2_cdf * (-r - 0.5 * vol * vol) / (vol * torch.sqrt(maturity)))
#
#     delta_t = (-spot * d1_cdf * (r + 0.5 * vol * vol) / (vol * torch.sqrt(maturity)) -
#                strike * torch.exp(-r * maturity) * d2_cdf * (-r - 0.5 * vol * vol) / (vol * torch.sqrt(maturity)))
#
#     delta_asset = d1_cdf
#
#     # Gamma (second derivative with respect to the spot price)
#     gamma_asset = d1_pdf / (spot * vol * torch.sqrt(maturity))
#
#     return delta_t, delta_asset, gamma_asset
#
#
# class loss_function(nn.Module):
#     def __init__(self, lambda_weight):
#         super(loss_function, self).__init__()
#         self.lambda_weight = lambda_weight
#
#     def forward(self, Y_hat, Y, params):
#         type_ = params[:, 0]
#         maturity = params[:, 1]
#         vol = params[:, 2]
#         strike = params[:, 3]
#         spot = params[:, 4]
#         r = params[:, 5]
#
#         # 使用BS方程计算期权价格
#         Y_BS = black_scholes_price(type_, spot, strike, maturity, vol, r)
#
#         # 计算Black-Scholes导数
#         delta_t, delta_asset, gamma_asset = black_scholes_derivatives(type_, spot, strike, maturity, vol, r)
#
#         # 计算MSE损失
#         loss_1 = nn.MSELoss(reduction='none')(Y_hat, Y)
#         # loss_1 = Y_hat - Y
#         # loss_1 = torch.norm(loss_1, p1, dim=0) / batch_size
#
#         # 计算新的损失项
#         additional_loss = r * Y_BS + delta_t - r * spot * delta_asset - 0.5 * vol**2 * spot**2 * gamma_asset
#         additional_loss = torch.norm(additional_loss, p=2, dim=0)
#
#         # 将新的损失项加权求和后添加到损失函数中
#         total_loss = loss_1 + self.lambda_weight * additional_loss
#         total_loss_mean = total_loss.mean()
#         total_loss_ = Variable(total_loss_mean, requires_grad=True)
#
#         # 返回平均损失
#         return total_loss_
