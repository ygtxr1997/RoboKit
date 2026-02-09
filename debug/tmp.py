import torch
import torch.nn as nn
import torch.optim as optim

# 简单的模型
model = nn.Sequential(
    nn.Linear(1000, 10000),
    nn.ReLU(),
    nn.Linear(10000, 1000),
).cuda()

# 随机输入数据
input_data = torch.randn(32, 1000).cuda()  # 使用GPU

# 损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 模拟训练过程
for _ in range(10000):  # 1000次训练
    output = model(input_data)
    loss = loss_fn(output, torch.randn(32, 1000).cuda())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
