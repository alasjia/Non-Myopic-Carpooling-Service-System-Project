'''
Created on Sunday December 04 2022
@author: lyj

基于DDPG框架，Actor和Critic神经网络均采用CNN
针对不明确的派车策略，采用简单多智能体，每辆车的action为调度函数中的附加值
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

# Actor Net
# Actor：输入是state，输出的是一个确定性的action       
#输入的state_batch形状为(32, 2, H = 22, W = 40,)，类型为torch.Tensor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        #卷积神经网络
        super(Actor,self).__init__()
        self.conv1 = nn.Sequential(      
            nn.Conv2d(in_channels=state_dim, out_channels=32, kernel_size=(4,8), stride=2, padding=1),
            nn.ReLU(inplace=True),             #output_size: H*W = 11*18
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5,6), stride=2, padding=1),
            nn.ReLU(inplace=True),              #output_size: H*W = 5*8
            nn.MaxPool2d(kernel_size=3, stride=1)  #output_size: H*W = 3*6
        )
        self.fc1 = nn.Sequential(
            nn.Linear(288, 8),        #input_size: 3*6*16 = 288
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(8, action_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)      # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）/x=x.squeeze()减少一个维度
        x = self.fc1(x)
        #控制action的取值范围
        # x = torch.tanh(x)    #（-1，1）
        action = self.out(x)
        return action

# Critic Net
# Critic：输入的除了当前的state还有Actor输出的action，然后输出的是Q-value       
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic,self).__init__()
        self.conv_s1 = nn.Sequential(
            nn.Conv2d(in_channels=state_dim, out_channels=32, kernel_size=(4,8), stride=2, padding=1),
            nn.ReLU(inplace=True),          #output_size: H*W = 11*18
        )
        self.conv_s2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5,6), stride=2, padding=1),
            nn.ReLU(inplace=True),                   #output_size: H*W = 5*8
            nn.MaxPool2d(kernel_size=(5, 6), stride= 2)      #output_size: H*W = 1*2
        )
        # self.fc_a = nn.Sequential(               
        #     nn.Linear(action_dim, 32),
        #     nn.ReLU(inplace=True),
        # )
        self.out = nn.Sequential(
            nn.Linear(36,1)        #通道数*Hp*Wp+aciton_dim = 16*1*2+4
        )

    #输入的s形状为(batch, 3, 22, 40)，a形状为(batch, 4)
    def forward(self,s,a):
        x = self.conv_s1(s)
        x = self.conv_s2(x)
        # y = self.fc_a(a)
        x = x.view(x.size(0), -1)      #(batch, channels, H, W) ---> (batch, channels*H*W)
        actions_value =  self.out(F.relu(torch.cat([x, a],dim = 1)))   #(batch, channels*H*W+aciton_dim)
        return actions_value
  

class DDPG(object):
    def __init__(self, state_dim, action_dim, replacement, memory_capacity, gamma=0.9, lr_a=0.001, lr_c=0.002, batch_size=32) :
        super(DDPG,self).__init__()
        # self.state_dim = state_dim         
        # self.action_dim = action_dim
        self.memory_capacity = memory_capacity            #记忆池容量
        self.replacement = replacement                    #target网络更新方式：软更新
        self.t_replace_counter = 0                        #硬更新时用于统计学习步数
        self.gamma = gamma                                #Target Q计算参数
        self.lr_a = lr_a                 #学习率
        self.lr_c = lr_c
        self.batch_size = batch_size     #小批量学习容量
        # 初始化记忆库
        #每一条记忆信息包括（st，at，st+1，rt），奖励：scalar  动作: (float) (num of car)  状态: (int) (3, num of node)
        #self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1)) 
        self.memory = deque()
        
        self.pointer = 0                     #存储每条数据的位置指针
        
        # 初始化Actor与Critic估计网络
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim,action_dim)
        
        # 初始化Actor与Critic目标网络，初始时目标网络与估计网络参数相同     
        self.actor_target = Actor(state_dim, action_dim)
        self.critic_target = Critic(state_dim,action_dim)
        
        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()
    
    # 从记忆库中随机采样    
    def sample(self):
        minibatch = random.sample(self.memory, self.batch_size)
        return minibatch
    
    #采用估计网络选择动作
    def choose_action(self, s):
        s = torch.from_numpy(s).float()
        #增加一个维度用于输入CNN
        s = torch.unsqueeze(s, dim=0)
        action = self.actor(s)
        return action.detach().numpy()   

    '''
    detach()：返回一个新的tensor，将action从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只
    是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
    '''

    def learn(self):
        # soft replacement and hard replacement
        # 用于更新target网络的参数
        
        if self.replacement['name'] == 'soft':
            # soft的意思是每次learn的时候更新部分参数
            tau = self.replacement['tau']
            a_layers = self.actor_target.named_children()       #返回actor_target网络的子模块迭代器
            c_layers = self.critic_target.named_children()
            for al in a_layers:
                 #x.mul_(y)：Torch里面所有带"_"的操作，都是in-place的，即存储到原来的x中。
                #值得注意的是，x必须是tensor, y可以是tensor，也可以是数。
                
                #al[0]是layer的名字【conv1、fc...】，al[1]是layer模块本身【Sequential()】，al[1][0]是Sequential()中的第一个layer
                # print(al[0])
                # print(al[1][0])
                # print(self.actor.state_dict())
                al[1][0].weight.data.mul_((1-tau))                   
                al[1][0].weight.data.add_(tau * self.actor.state_dict()[al[0]+'.0.weight'])  #0：因为是conv1的第一层
                al[1][0].bias.data.mul_((1-tau))
                al[1][0].bias.data.add_(tau * self.actor.state_dict()[al[0]+'.0.bias'])
            for cl in c_layers:
                cl[1][0].weight.data.mul_((1-tau))
                cl[1][0].weight.data.add_(tau * self.critic.state_dict()[cl[0]+'.0.weight'])
                cl[1][0].bias.data.mul_((1-tau))
                cl[1][0].bias.data.add_(tau * self.critic.state_dict()[cl[0]+'.0.bias'])
            
        else:                 #???报错和卷积网络有关
            # hard的意思是每隔一定的步数才更新全部参数
            if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    al[1][0].weight.data = self.actor.state_dict()[al[0]+'.0.weight']
                    al[1][0].bias.data = self.actor.state_dict()[al[0]+'.0.bias']
                for cl in c_layers:
                    cl[1][0].weight.data = self.critic.state_dict()[cl[0]+'.0.weight']
                    cl[1][0].bias.data = self.critic.state_dict()[cl[0]+'.0.bias']
            
            self.t_replace_counter += 1

        # 从记忆库中采样bacth data
        batch = self.sample()
        state_batch = [data[0] for data in batch]      #'list'   'numpy.float64' 和下述一致
        action_batch = [data[1] for data in batch]
        reward_batch = [data[2] for data in batch]
        state_batch_ = [data[3] for data in batch]

        #将batch中的list数据类型转为torch.Tensor
        state_batch = torch.from_numpy(np.array(state_batch).astype(np.float32))      #(32, 2, 22, 40）print(state_batch_.shape) 
        state_batch_ = torch.from_numpy(np.array(state_batch_).astype(np.float32))
        action_batch = torch.from_numpy( np.array(action_batch).astype(np.float32))            #(32,4) 
        reward_batch = torch.from_numpy(np.reshape(np.array(reward_batch), (32,1)).astype(np.float32))      #(32, 1)  


        # 训练Actor（估计网络）
        #估计actor根据st获得估计动作，估计critic根据st和估计动作获得q；
        a = self.actor(state_batch)        
        q = self.critic(state_batch, a)
        #梯度上升更新actor（估计）的参数
        a_loss = -torch.mean(q)
        self.aopt.zero_grad()
        a_loss.backward(retain_graph=True)       #要想对某个变量重复调用backward，则需要将该参数设置为 True
        self.aopt.step()
        
        # 训练critic（估计网络）
        #目标actor根据st+1获得目标动作，目标critic根据st+1、目标动作、rt获得目标q值：获取对“最好动作”的评价
        a_ = self.actor_target(state_batch_)
        q_ = self.critic_target(state_batch_, a_)
        q_target = reward_batch + self.gamma * q_
        #估计critic根据st和实际动作（由估计actor输出的）获得估计q值：估计critic对实际做出的动作进行评价
        q_eval = self.critic(state_batch, action_batch)
        #梯度下降更新
        td_error = self.mse_loss(q_target,q_eval)
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()

    # 存储序列数据
    def store_transition(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))
        if len(self.memory) > self.memory_capacity:
            self.memory.popleft()
        self.pointer  +=  1

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


if __name__ == '__main__':
    # hyper parameters
    NUM_CAR = 500       #车辆数
    # NEAR_CARS = []           #附近车辆编号数组  
    NUM_NODE = 880    #站点数
    H_Grid = 22              #地理网格高度
    W_Grid = 40             #地理网格宽度
    Num_State = 3     #状态数量
    # NUM_REQ = 100     #请求数
    # NUM_TASK = NUM_REQ * 2   #任务数
    MAX_EPISODES = 100       #运行次数（天数）
    MAX_EP_STEPS = 1400      #每个episode的步数，时间间隔初设为1分钟，请求发出持续到1440分钟，即1440step   
    MEMORY_CAPACITY = 200000   #记忆池容量
    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=200)
    ][0]  # you can try different target replacement strategies

    #设置环境：输入为目标调度函数的权重系数，输出为奖励、下一时刻的状态、是否结束（？）
    ll = ctypes.cdll.LoadLibrary
    env = ll(r"/home/chwei/JeanLu/SharedTaxi/C_python/CP1214/libPolicy1_cnn.so")  

    # create the arrays for the input of C's functions
    Car_Distribution = np.empty((NUM_NODE), dtype = float)   #OUTPUT: global state and car state
    Req_Distribution = np.empty((NUM_NODE), dtype = float)   
    Car_Location = np.empty((NUM_NODE), dtype = float)
    Reward = np.zeros(1, dtype = float)

    # convert the numpy arrays or variables to the data type in C
    p2c = ConvertP2C()
    Car_Distribution = p2c.ConvertListFlt(Car_Distribution) 
    Req_Distribution = p2c.ConvertListFlt(Req_Distribution)
    Car_Location = p2c.ConvertListFlt(Car_Location)
    Reward = p2c.ConvertListFlt(Reward)
    

    #【状态】地理网格图——请求分布、车辆分布 + 单个车辆位置；
    s_dim =  Num_State    
    #【动作】一维数组：调度函数中附加值
    a_dim = 1       

    ddpg = DDPG(state_dim=s_dim,
                action_dim=a_dim,
                replacement=REPLACEMENT,
                memory_capacity=MEMORY_CAPACITY)

    t1 = time.time()
    for ep in range(MAX_EPISODES):
        # 计算每个episode运行时间
        t_ep = time.time()
        #initialize all states in the beginning of the episode, including states of request, car and network
        env.sys_reset(Car_Distribution, Req_Distribution, Car_Location)  ???
        #初始化episode的累计奖励
        ep_reward = 0       
        #初始化所有车辆的权重系数(0.5)和所有车辆独立状态
        a_all = np.zeros(a_dim, dtype = float)
        s_car_all =  np.zeros((NUM_CAR, NUM_NODE), dtype = float)
        # control exploration   标准差
        VAR = 0.1
        
        #遍历每一个时间步
        for t in range(MAX_EP_STEPS):     
            # 初始化二维状态数组
            CarDistribution_2D = np.zeros((H_Grid, W_Grid,), dtype = float)
            ReqDistribution_2D = np.zeros((H_Grid, W_Grid), dtype = float)
            CarLocation_2D = np.zeros((H_Grid, W_Grid), dtype = float)
            rows = -1
            #将Car_Distribution一维状态【880，】表示转为二维【22，40】
            for i in range(NUM_NODE):                
                if (i % 40 == 0):
                    rows += 1
                CarDistribution_2D[rows][i % 40] = np.array(Car_Distribution)[i]
                ReqDistribution_2D[rows][i % 40] = np.array(Req_Distribution)[i]
                CarLocation_2D[rows][i % 40] = np.array(CarLocation_2D)[i]
                # ReqDistribution_all[rows][i % 40] += np.array(Req_Distribution)[i]
            #获取全局共享状态：路网中车辆分布和请求分布
            s_global = np.stack((CarDistribution_2D, ReqDistribution_2D), axis = 0)   #(2, 22, 40)       
            # print(ReqDistribution_2D)
            # print(ReqDistribution_2D.shape)
            #遍历所有车辆
            for car in range(NUM_CAR):       
                #输入车辆编号，获得车辆状态：目前仅为车辆位置 DDPG    (车辆位置、空座数、路径)
                car_id = p2c.ConvertInt(car)
                env.car_state_generate(car_id, Car_Location) 
                #存储车辆状态
                s_car = np.array(Car_Location)  ???
                s_car_all[car] = s_car                            
                #拼接全局状态和车辆状态，获得全部状态           
                s = np.append(s_global, s_car)  #(75,)    'numpy.float32'
                #为该车辆选择权重系数
                a = ddpg.choose_action(s)     #返回的a为numpy.float()
                a = a.astype(np.float32)
                #在动作选择上添加随机噪声  （在合乘调度中，权重系数的取值范围在0-1之间）
                #clip函数防止添加噪声后超出[0,1]
                # a = np.clip(np.random.normal(a, VAR), 0, 10)  
                #存储该车辆的权重系数
                a_all[car] = a
            
            #获取动作值，并转换数据类型为float
            a_all = ddpg.choose_action(s_global)     #返回的a为numpy.float()
            a_all = np.squeeze(a_all, axis = 0)      #(1, 4)  ---->  (4,)
            a_all= a_all.astype(np.float32)             
            #将动作丢入环境获取奖励和下一时刻全局状态
            interval = p2c.ConvertInt(t)
            a_all_c = p2c.ConvertListFlt(a_all)
            env.execution(a_all_c, interval, Car_Distribution, Req_Distribution, Car_Location, Reward)

            r = np.array(Reward)[0]        

            #将一维状态表示转为二维
            CarDistribution_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)
            ReqDistribution_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)
            rows = -1
            for i in range(NUM_NODE):                
                if (i % 40 == 0):
                    rows += 1
                CarDistribution_2D_[rows][i % 40] = np.array(Car_Distribution)[i]
                ReqDistribution_2D_[rows][i % 40] = np.array(Req_Distribution)[i]
            #获取t+1全局共享状态：路网中车辆分布和请求分布
            s_global_ = np.stack((CarDistribution_2D_, ReqDistribution_2D_), axis = 0)   #(2,22,40)       
            # print(s_global_.shape)
            #done =    ???
            #info     ???
            #存储数据 在记忆池
            ddpg.store_transition(s_global, a_all, r, s_global_)          #s:(2,22,40)   a_all[car]:scalar    r: scalar     s_:(2,22,40)

            if ddpg.pointer > MEMORY_CAPACITY: 
                VAR *= .9995  # decay the action randomness，逐步减少噪声，使动作选择越来越确定
                ddpg.learn()
                print(a_all)  
                if t==0:
                    print(" MEMORY_CAPACITY is full!!!")

            #检查每辆车的action
            # print(a_all)

            #更新状态
            s_global = s_global_
            #更新累计奖励
            ep_reward += r 
            #如果达到了最大步数（一天结束），打印奖励和噪声大小，并退出循环
            if t == MAX_EP_STEPS - 1:
                print('Episode:', ep, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % VAR, )
                print('Running time: ', time.time() - t_ep) 
                break

    print('Running time: ', time.time() - t1) 






