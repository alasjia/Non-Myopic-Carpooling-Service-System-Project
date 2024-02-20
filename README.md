# SharedTaxiProject
 
实验环境：pytorch、Anoconda、linux ubantu18.04

**1、深度神经网络的设置**
分别测试全连接神经网络(FC)和卷积神经网络（CNN）；
注意：
（1）在网络参数更新过程中，发现layers中的激活函数（ReLU）和池化层（MaxPool2d）无参数，即只有nn.Conv2d()的参数需要更新。

**2、智能体和参数学习**
对于离散型和连续型动作空间分别测试DQN和DDPG框架；
注意：
（1）这里设置的Q为【从t时刻开始，未来无限制的时间里系统总收益（回报）的估计值】
（2）对于batch size条经验数据分类时，即
state_batch = torch.tensor(transition_dict['states'], dtype = torch.float32).to(self.device)
torch.tensor()将numpy数据形式转为torch形式，.to(self.device)将state_batch放置在GPU上

**3、经验回放池的设置**
这部分采用了deque()队列的形式存放经验，用字典形式存放batch size的经验。也可以用list存放batch size的经验，个人觉得字典更好。

**4、训练过程**
注意超参数的设置；
设置在GPU上运算；
设置环境，初始化agent、replay buffer等；

**5、网络参数存储与重载**
存储与重载模型参数：存储参数用到state_dict()，重载参数用到load_state_dict()，注意这种方法只保存网络参数，不保存网络结构。

**6、可视化**
一般来说，希望检查的内容包括Reward、Loss（Q和Target Q的差值）。
