mlp部分代码更改
model.py
1. 第78行加入y=y.long()以解决一些bug
main.py
1. 第15行import json用于存储训练结果（loss和accuracy信息）
2. 第102-105行定义一些列表用于存储训练结果
3. 第155-158行在列表中插入得到的loss和accuracy
4. 第166-173行把这些数据存储到文件里，便于之后画图