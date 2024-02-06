cnn部分代码更改
model.py
1. 第8行加入import torch.cuda，便于在cuda上生成随机tensor作为dropout layer中的mask
2. 第91行加入y=y.long()以解决一些bug
main.py
1. 第14行把load_cifar_2d改为load_cifar_4d，更正错误
2. 第15行import json用于存储训练结果（loss和accuracy信息）
3. 第102-105行定义一些列表用于存储训练结果
4. 第155-158行在列表中插入得到的loss和accuracy
5. 第166-173行把这些数据存储到文件里，便于之后画图