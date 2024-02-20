'''
Created on Friday Feb 24 2023
@author: lyj

基于DQN框架，实现对出租车合乘调度系统策略的学习
Q网络采用CNN
动作空间为离散型，包括三种调度策略
相较于0212改变了状态表征：过去请求分布、当前可用车辆分布、新请求的上下车位置
（过去请求分布、过去可用车辆分布、新请求上车位置为中心的上车分布）

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctypes
import time
from collections import deque
import random

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)


#Q network
class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        self.conv1 = nn.Sequential(      
            nn.Conv2d(in_channels=state_dim, out_channels=16, kernel_size=(3,3), stride=1, padding=0),
            nn.ReLU(inplace=True),             #output_size: H*W = 20*38*32
            nn.MaxPool2d(kernel_size=3, stride=1)  #output_size: H*W = 18*36
        )
        self.conv2 = nn.Sequential(      
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=0),
            nn.ReLU(inplace=True),             #output_size: H*W = 20*38*32
            # nn.MaxPool2d(kernel_size=3, stride=1)  #output_size: H*W = 18*36
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=1, padding=0),
            nn.ReLU(inplace=True),              #output_size: H*W = 14*32*64
            # nn.MaxPool2d(kernel_size=5, stride=1)  #output_size: H*W = 10*28
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,5), stride=1, padding=0),
            nn.ReLU(inplace=True),              #output_size: H*W = 8*24*128
            nn.MaxPool2d(kernel_size=3, stride=1)  #output_size: H*W = 6*22
        )
        self.fc1 = nn.Sequential(
            nn.Linear(24576, 128),        #input_size: 6*22*128=
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)      # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）/x=x.squeeze()减少一个维度
        x = self.fc1(x)
        action = self.out(x)
        return action

class DQN(object):
    def __init__(self, state_dim, action_dim, epsilon, memory_capacity, gamma, learning_rate, batch_size, target_update, device) :
        super(DQN,self).__init__()
        self.state_dim = state_dim         
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.memory_capacity = memory_capacity            #记忆池容量
        self.gamma = gamma                                #Target Q计算参数
        self.learning_rate = learning_rate                 #学习率
        self.batch_size = batch_size     #小批量学习容量
        # 初始化记忆库
        #每一条记忆信息包括（st，at，st+1，rt），奖励：scalar  动作: (float) (num of car)  状态: (int) (3, num of node)
        self.memory = deque()
        
        self.pointer = 0                     #存储每条数据的位置指针
        self.target_update = target_update    #目标网络更新频率
        self.count = 0               #计数器，记录网络更新次数
        self.device = device
        
        # 初始化估计网络
        self.q_net = Qnet(state_dim, action_dim).to(device)
        # 初始化目标网络，初始时目标网络与估计网络参数相同     
        self.q_net_target = Qnet(state_dim, action_dim).to(device)
        
        # 定义优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  # 选取损失函数
        self.mse_loss = nn.MSELoss()

    # 从记忆库中随机采样    
    def sample(self):
        minibatch = random.sample(self.memory, self.batch_size)
        return minibatch

    # 存储序列数据
    def store_transition(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))
        if len(self.memory) > self.memory_capacity:
            self.memory.popleft()
        self.pointer  +=  1

    #  epsilion-贪婪策略采取动作
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim, dtype = "int64")
        else:
            state = torch.from_numpy(state).float()
            #增加一个维度用于输入CNN
            state = torch.unsqueeze(state, dim=0).to(self.device)
            action = self.q_net(state).argmax().cpu().numpy()  #选择最大值对应的序号【0、1、2】;如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式
        return action

    def learn(self):
        #可改进：用字典建立记忆库，list比较麻烦
        # 从记忆库中采样bacth data
        batch = self.sample()
        state_batch = [data[0] for data in batch]      #'list'   'numpy.float64' 和下述一致
        action_batch = [data[1] for data in batch]
        reward_batch = [data[2] for data in batch]
        state_batch_ = [data[3] for data in batch]

        #将batch中的list数据类型转为torch.Tensor
        state_batch = torch.from_numpy(np.array(state_batch).astype(np.float32)).to(self.device)      #(32, 2, 22, 40）print(state_batch_.shape) 
        state_batch_ = torch.from_numpy(np.array(state_batch_).astype(np.float32)).to(self.device)
        action_batch = torch.from_numpy( np.array(action_batch)).view(-1,1).to(self.device)            #(32,1)  int64
        reward_batch = torch.from_numpy(np.reshape(np.array(reward_batch), (self.batch_size,1)).astype(np.float32)).to(self.device)      #(32, 1)  
        # 增加维度：view(-1,1)是tprch的方法，reshape()是numpy的方法，前者可自动计算

        # 训练Q network
        q_values = self.q_net(state_batch).gather(dim = 1, index = action_batch)  #dim=1, index=[5,3] 相当于取第0行中第5个、第1行中第3个
        max_next_q_values = self.q_net_target(state_batch_).max(1)[0].view(-1, 1)     #view(-1,1)将(batch,)变为(batch,1)
        q_targets = reward_batch + self.gamma * max_next_q_values     #TD误差目标
        dqn_loss = torch.mean(self.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad   #Pytorch中默认梯度会累积，这里需要显式将梯度设置为0
        dqn_loss.backward()  #反向传播更新参数
        self.optimizer.step()

        #延迟更新目标网络
        if self.count % self.target_update == 0:
            self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.count += 1


#Convert the python data type into C
class ConvertP2C(object):            
    def ConvertListFlt(self,a):
        return (ctypes.c_float * len(a))(*a)   

    def ConvertListDbl(self, a):
        return (ctypes.c_double * len(a))(*a)  

    def ConvertDouble(self, num):
        return ctypes.c_double(num)

    def ConvertFloat(self, num):
        return ctypes.c_float(num)

    def ConvertInt(self, num):
        return ctypes.c_int(num)


# 存储模型参数
def save_parameters():
    #存储
    qnet_paras = {"q_eval_net": dqn.q_net.state_dict(), "q_targ_net": dqn.q_net_target.state_dict(), "qnet_Adam": dqn.optimizer.state_dict(), "epoch": ep}
    torch.save(obj=qnet_paras, f="CP0212/qnet_paras_save.pth")

# 重载模型参数
def reload_parameters():
    #state_dict() 只保存网络参数，不保存网络结构
    reload_states = torch.load("CP0212/qnet_paras_save.pth")#重载的reload_states 为一个字典
    dqn.q_net.load_state_dict(reload_states["q_eval_net"]) 
    dqn.q_net_target.load_state_dict(reload_states['q_targ_net'])
    dqn.optimizer.load_state_dict(reload_states['qnet_Adam'])


if __name__ == '__main__':
    # hyper parameters
    NUM_CAR = 500       #车辆数
    NUM_NODE = 880    #站点数
    H_Grid = 22              #地理网格高度
    W_Grid = 40             #地理网格宽度
    MAX_EPISODES = 50           #运行次数（天数）
    MAX_EP_STEPS = 700    #每个episode的步数，时间间隔初设为1分钟，请求发出持续到1440分钟，即1440step   
    memory_capacity = 7000   #记忆池容量，20000
    lr = 0.001
    gamma = 0.98
    epsilion = 0.01      #可改进：可以进行衰减
    target_update  = 10
    batch_size = 32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #【状态】两张地理网格图——请求分布、车辆分布；
    s_dim =  6    
    #【动作】一维数组：调度函数中，乘客数分别为0，1，2，3时的附加值
    a_dim = 3       
    #设置环境：输入为目标调度函数的权重系数，输出为奖励、下一时刻的状态、是否结束（？）
    ll = ctypes.cdll.LoadLibrary
    env = ll(r"/home/chwei/JeanLu/SharedTaxi/C_python/CP0224/libPolicy2_0227.so")  

    # create the arrays for the input of C's functions
    CarChange_Distribution = np.empty((NUM_NODE), dtype = float)  
    Req_Distribution_15 = np.empty((NUM_NODE), dtype = float)   
    Seat_Distribution_Req = np.empty((NUM_NODE), dtype = float)  
    Reward = np.zeros(1, dtype = float)

    # convert the numpy arrays or variables to the data type in C
    p2c = ConvertP2C()
    CarChange_Distribution = p2c.ConvertListFlt(CarChange_Distribution)
    Req_Distribution_15 = p2c.ConvertListFlt(Req_Distribution_15)
    Seat_Distribution_Req = p2c.ConvertListFlt(Seat_Distribution_Req)
    Reward = p2c.ConvertListFlt(Reward)

# 设置agent
    dqn = DQN(state_dim=s_dim,
                action_dim=a_dim,
                epsilon = epsilion,
                memory_capacity=memory_capacity,
                gamma = gamma,
                learning_rate = lr,
                batch_size = batch_size,
                target_update = target_update,
                device = device)

    # # 如需继续训练，重载参数
    # reload_parameters()

    # 记录开始时间
    t1 = time.time()
    # 确定曼哈顿区域内站点ID
    env.nodes_in_manhattan_fixed();

    #开始训练，记忆池满后开始参数更新
    for ep in range(MAX_EPISODES):
        # 计算每个episode运行时间
        t_ep = time.time()
        #initialize all states in the beginning of the episode, including states of request, car and network
        env.sys_reset(CarChange_Distribution, Req_Distribution_15, Seat_Distribution_Req)
        #初始化episode的累计奖励
        ep_reward = 0       

        # env.input_check();
        
        #遍历每一个时间步
        for t in range(MAX_EP_STEPS):     
            # 初始化二维状态数组
            CarChange_2D = np.zeros((H_Grid, W_Grid), dtype = float)
            ReqDistribution15_2D = np.zeros((H_Grid, W_Grid), dtype = float)
            Seat_Distribution_2D = np.zeros((H_Grid, W_Grid), dtype = float)
            rows = -1
            #将Car_Distribution一维状态【880，】表示转为二维【22，40】
            for i in range(NUM_NODE):                
                if (i % 40 == 0):
                    rows += 1
                CarChange_2D[rows][i % 40] = np.array(CarChange_Distribution)[i]
                ReqDistribution15_2D[rows][i % 40] = np.array(Req_Distribution_15)[i]
                Seat_Distribution_2D[rows][i % 40] = np.array(Seat_Distribution_Req)[i]
            #获取全局共享状态：路网中车辆分布和请求分布
            s_global = np.stack((CarChange_2D, ReqDistribution15_2D, Seat_Distribution_2D), axis = 0)   #(3, 22, 40)       
            # # check
            # print("--------------------step", t, "--------------------------")
            # for i in range(NUM_NODE):
            #     if(np.array(Car_Distribution_30)[i] != 0):
            #         print(i, ":", np.array(Car_Distribution_30)[i])
            # print(s_global.shape)
            # print(ReqDistribution_2D.shape)
            
            #获取动作值，并转换数据类型为float
            a = dqn.take_action(s_global)        #返回的a为numpy.int


            #将动作丢入环境获取奖励和下一时刻全局状态
            t_c = p2c.ConvertInt(t)
            a_c = p2c.ConvertInt(a)
            env.execution(a_c, t_c, CarChange_Distribution, Req_Distribution_15, Seat_Distribution_Req, Reward)

            r = np.array(Reward)[0] 
            # print("step%d reward:%f"%(t, r))     

            #t+1时刻状态，将一维状态表示转为二维
            CarChange_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)
            ReqDistribution15_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)
            Seat_Distribution_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)

            rows = -1
            for i in range(NUM_NODE):                
                if (i % 40 == 0):
                    rows += 1
                CarChange_2D_[rows][i % 40] = np.array(CarChange_Distribution)[i]
                ReqDistribution15_2D_[rows][i % 40] = np.array(Req_Distribution_15)[i]
                Seat_Distribution_2D_[rows][i % 40] = np.array(Seat_Distribution_Req)[i]
            #获取t+1全局共享状态：路网中车辆分布和请求分布
            s_global_ = np.stack((CarChange_2D_, ReqDistribution15_2D_, Seat_Distribution_2D_), axis = 0)   #(3,22,40)       
            # print("time id:", t)
            # print(ReqDistribution30_2D_)
            #存储数据 在记忆池
            dqn.store_transition(s_global, a, r, s_global_)          #s:(2,22,40)   a[car]:scalar    r: scalar     s_:(2,22,40)
            # print("interval", t, ":", r)

            if dqn.pointer > memory_capacity: 
                dqn.learn()
                if t==0:
                    print("action in t=0: ",a)  
                elif t==10:
                    print("action in t=10: ",a)  
                elif t==20:
                    print("action in t=20: ",a)  
                elif t==30:
                    print("action in t=30: ",a)  
                elif t==50:
                    print("action in t=50: ",a)  
                elif t==100:
                    print("action in t=100: ",a)  
                elif t==200:
                    print("action in t=200: ",a)  
                elif t==300:
                    print("action in t=300: ",a)  
                elif t==400:
                    print("action in t=400: ",a)  
                else:
                    pass
                # if t==0:
                #     print(" MEMORY_CAPACITY is full!!!")

            #检查每步系统采取第几类调度策略
            # print(a)

            #更新状态
            s_global = s_global_
            #更新累计奖励
            ep_reward += r 
            # print("ep_reward: %f"%ep_reward)
            #如果达到了最大步数（一天结束），打印奖励和噪声大小，并退出循环
            if t == MAX_EP_STEPS - 1:
                print('Episode:', ep, ' Reward: %i' % int(ep_reward))
                print('Running time: ', time.time() - t_ep) 
                break

    print('Running time: ', time.time() - t1) 
    
    # 存储参数，方便继续训练
    save_parameters()





