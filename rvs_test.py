from torch import nn
from torch.nn import Sequential, Linear
from dataset import *

# 定义训练的设备
device = torch.device("cuda")


class Rvs(nn.Module):
    def __init__(self):
        super(Rvs, self).__init__()
        self.model1 = Sequential(
            Linear(120, 480),
            nn.ReLU(),
            Linear(480, 960),
            nn.ReLU(),
            Linear(960, 500),
            nn.ReLU(),
            Linear(500, 200),
            nn.ReLU(),
            Linear(200, 64),
            nn.ReLU(),
            Linear(64, 3)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = torch.load("newrvs_4001e-4.pth")
while(True):
    num=eval(input("Please input an integer between 1 and 41219."))
    curve=data[num-1]
    curve=curve.to(torch.float32)
    # print(data)
    output = model(curve)
    print("The predicted values are：{}".format(output))

    parameter=para[num-1]
    print("The real values are：     {}".format(parameter))
    print("--------------------------------------------------------------")
