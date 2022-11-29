import torch
import numpy as np

# 定义训练的设备
device = torch.device("cuda")

# 处理曲线数据
data0S11 = np.genfromtxt('C_data/0S11.csv', delimiter=",")
data0S11 = torch.from_numpy(data0S11)
data0S11 = data0S11.to(device)
data0S11 = data0S11[:, 1]
data2S11 = np.genfromtxt('C_data/2S11.csv', delimiter=",")
data2S11 = torch.from_numpy(data2S11)
data2S11 = data2S11.to(device)
data2S11 = data2S11[:, 1]
data4S11 = np.genfromtxt('C_data/4S11.csv', delimiter=",")
data4S11 = torch.from_numpy(data4S11)
data4S11 = data4S11.to(device)
data4S11 = data4S11[:, 1]
data5S11 = np.genfromtxt('C_data/5S11.csv', delimiter=",")
data5S11 = torch.from_numpy(data5S11)
data5S11 = data5S11.to(device)
data5S11 = data5S11[:, 1]
data6S11 = np.genfromtxt('C_data/6S11.csv', delimiter=",")
data6S11 = torch.from_numpy(data6S11)
data6S11 = data6S11.to(device)
data6S11 = data6S11[:, 1]
data7S11 = np.genfromtxt('C_data/7S11.csv', delimiter=",")
data7S11 = torch.from_numpy(data7S11)
data7S11 = data7S11.to(device)
data7S11 = data7S11[:, 1]
data = torch.cat((data0S11, data2S11, data4S11, data5S11, data6S11, data7S11), 0)
data = data.to(device)
data = data.reshape(41219, 120)

# 处理参数数据
para0 = np.genfromtxt('C_data/rparameter0.txt')
para0 = torch.from_numpy(para0)
para0 = para0.to(device)
para2 = np.genfromtxt('C_data/rparameter2.txt')
para2 = torch.from_numpy(para2)
para2 = para2.to(device)
para4 = np.genfromtxt('C_data/rparameter4.txt')
para4 = torch.from_numpy(para4)
para4 = para4.to(device)
para5 = np.genfromtxt('C_data/rparameter5.txt')
para5 = torch.from_numpy(para5)
para5 = para5.to(device)
para6 = np.genfromtxt('C_data/rparameter6.txt')
para6 = torch.from_numpy(para6)
para6 = para6.to(device)
para7 = np.genfromtxt('C_data/rparameter7.txt')
para7 = torch.from_numpy(para7)
para7 = para7.to(device)
para = torch.cat((para0, para2, para4, para5, para6, para7), 0)
para = para.to(device)
para = para.reshape(41219, 3)

