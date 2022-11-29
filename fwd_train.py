from torch import nn
from torch.nn import Linear, Sequential
from torch.utils.data import DataLoader

from fwd_load_dataset import *

train_data = fwd_dataset()
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
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


# 创建网络模型
model = Fwd()
model = model.to(device)
# 损失函数
loss_fn = nn.MSELoss()
loss_fn = loss_fn.to(device)
# 优化器
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 记录训练轮数
epoch = 50


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 1500 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


for t in range(epoch):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    print("Done!")
    # torch.save(model,"newfwd_{}lr1e-3.pth".format(epoch))
