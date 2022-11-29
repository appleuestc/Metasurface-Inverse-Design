from tkinter import *

import matplotlib.pyplot as plt
from torch import nn
from torch.nn import Sequential, Linear

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


def testForward(event=None):
    parameter = para[eval(num.get()) - 1]
    parameter = parameter.to(torch.float32)
    output = model(parameter)
    output = output.detach().cpu().numpy()
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(1200, 300, 640, 545)
    plt.title("Forward Result")
    plt.xlabel("Frequency")
    plt.plot(output)
    curve = data[eval(num.get()) - 1]
    curve = curve.detach().cpu().numpy()
    plt.plot(curve)
    plt.show()


model = torch.load("newfwd_450lr1e-4.pth")
root = Tk()
root.geometry("640x100")
root.title("Welcome!")
frame = Frame(root)
frame.pack()
# label = Label(frame, text="有41219组数据，请问您想验证第几组数据？", fg="black", font=("Arial", 12))
label = Label(frame, text="Please input an integer between 1 and 41219.", fg="black", font=("Arial", 12))
label.pack()
num = Entry(frame, width=64)
num.pack(padx=5, pady=5)

Button = Button(frame, text="Submit", command=testForward, font=("Arial", 12))
# Button.bind('<Return>', testForward)
Button.pack(padx=5, pady=5)
root.bind('<Return>', lambda event=None: Button.invoke())
root.mainloop()



