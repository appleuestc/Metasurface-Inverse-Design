from torch.utils.data import Dataset
from dataset import *
# 定义训练的设备
device = torch.device("cuda")

class fwd_dataset(Dataset):
    def __init__(self):
        # 处理曲线数据，每120个数作为一个输出
        self.data=data.float()
        print(data.shape)

        # 处理参数数据，每三个作为一个输入
        self.para=para.float()
        print(para.shape)

    def __len__(self):
        return 41219

    def __getitem__(self, idx):
        return self.para[idx], self.data[idx]
