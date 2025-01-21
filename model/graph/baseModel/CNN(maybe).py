import torch
import torch.nn.functional as F #使用functional中的ReLu激活函数

#CNN模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #两个卷积层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)  #1为in_channels 10为out_channels
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        #池化层
        self.pooling = torch.nn.MaxPool2d(2)  #2为分组大小2*2
        #全连接层 320 = 20 * 4 * 4
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        #先从x数据维度中得到batch_size
        batch_size = x.size(0)
        #卷积层->池化层->激活函数
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  #将数据展开，为输入全连接层做准备
        x = self.fc(x)
        return x
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  #将模型的所有内容放入cuda中

#设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
#神经网络已经逐渐变大，需要设置冲量momentum=0.5
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#训练
#将一次迭代封装入函数中
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):   #在这里data返回输入:inputs、输出target
        inputs, target = data
        #在这里加入一行代码，将数据送入GPU中计算！！！
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()

        #前向 + 反向 + 更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))

