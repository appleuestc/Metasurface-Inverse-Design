import matplotlib.pyplot as plt
from torch import nn
from torch.nn import Sequential, Linear
import numpy as np

from dataset import *

# 定义训练的设备
device = torch.device("cuda")


class Fwd(nn.Module):
    def __init__(self):
        super(Fwd, self).__init__()
        self.model1 = Sequential(
            Linear(3, 40),
            nn.ReLU(),
            Linear(40, 160),
            nn.ReLU(),
            Linear(160, 320),
            nn.ReLU(),
            Linear(320, 640),
            nn.ReLU(),
            Linear(640, 1280),
            nn.ReLU(),
            Linear(1280, 2560),
            nn.ReLU(),
            Linear(2560, 1280),
            nn.ReLU(),
            Linear(1280, 640),
            nn.ReLU(),
            Linear(640, 320),
            nn.ReLU(),
            Linear(320, 120)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = torch.load("newfwd_450lr1e-4.pth")

# a1 = np.random.randint(1, 41219)
# print(a1)
# a2 = np.random.randint(1, 41219)
# print(a2)
# a3 = np.random.randint(1, 41219)
# print(a3)
# a4 = np.random.randint(1, 41219)
# print(a4)
# a5 = np.random.randint(1, 41219)
# print(a5)
# a6 = np.random.randint(1, 41219)
# print(a6)

a1=35083
a2=23540
a3=39934
a4=39902
a5=39754
a6=14795

parameter1 = para[a1 - 1]
print(parameter1)
parameter1 = parameter1.to(torch.float32)
output1 = model(parameter1)
output1 = output1.detach().cpu().numpy()

curve1 = data[a1 - 1]
curve1 = curve1.detach().cpu().numpy()

plt.subplot(3, 2, 1)
plt.plot(output1)
plt.plot(curve1)
plt.legend(["predict","real"])
plt.title("plot 1")
# plot2
parameter2 = para[a2 - 1]
print(parameter2)
parameter2 = parameter2.to(torch.float32)
output2 = model(parameter2)
output2 = output2.detach().cpu().numpy()

curve2 = data[a2 - 1]
curve2 = curve2.detach().cpu().numpy()

plt.subplot(3, 2, 2)
plt.plot(output2)
plt.plot(curve2)
plt.legend(["predict","real"])
plt.title("plot 2")
# plot3
parameter3 = para[a3 - 1]
print(parameter3)
parameter3 = parameter3.to(torch.float32)
output3 = model(parameter3)
output3 = output3.detach().cpu().numpy()

curve3 = data[a3 - 1]
curve3 = curve3.detach().cpu().numpy()

plt.subplot(3, 2, 3)
plt.plot(output3)
plt.plot(curve3)
plt.legend(["predict","real"])
plt.title("plot 3")
# plot4
parameter4 = para[a4 - 1]
print(parameter4)
parameter4 = parameter4.to(torch.float32)
output4 = model(parameter4)
output4 = output4.detach().cpu().numpy()

curve4 = data[a4 - 1]
curve4 = curve4.detach().cpu().numpy()

plt.subplot(3, 2, 4)
plt.plot(output4)
plt.plot(curve4)
plt.legend(["predict","real"])
plt.title("plot 4")
# plot5
parameter5 = para[a5 - 1]
print(parameter5)
parameter5 = parameter5.to(torch.float32)
output5 = model(parameter5)
output5 = output5.detach().cpu().numpy()

curve5 = data[a5 - 1]
curve5 = curve5.detach().cpu().numpy()

plt.subplot(3, 2, 5)
plt.plot(output5)
plt.plot(curve5)
plt.legend(["predict","real"])
plt.title("plot 5")
# plot6
parameter6 = para[a6 - 1]
print(parameter6)
parameter6 = parameter6.to(torch.float32)
output6 = model(parameter6)
output6 = output6.detach().cpu().numpy()

curve6 = data[a6 - 1]
curve6 = curve6.detach().cpu().numpy()

plt.subplot(3, 2, 6)
plt.plot(output6)
plt.plot(curve6)
plt.legend(["predict","real"])
plt.title("plot 6")

plt.show()
