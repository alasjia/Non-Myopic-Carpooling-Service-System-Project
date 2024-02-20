'''
Created on Monday Mar 13 2023
@author: lyj

在DQN_CNN_old.py基础上改进：
优化replay buffer，将list改为dictionary；
增加可视化；
采用reinforce算法

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctypes
import time
from collections import deque
import random
import matplotlib.pyplot as plt

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Sequential(      
            nn.Conv2d(in_channels=state_dim, out_channels=8, kernel_size=(3,3), stride=1, padding=0),
            nn.ReLU(inplace=True),             #output_size: H*W = 20*38*8
            # nn.MaxPool2d(kernel_size=3, stride=1)  #output_size: H*W = 18*36
        )
        self.conv2 = nn.Sequential(      
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5,5), stride=1, padding=0),
            nn.ReLU(inplace=True),             #output_size: H*W = 16*34*16
            # nn.MaxPool2d(kernel_size=3, stride=1)  #output_size: H*W = 18*36
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), stride=1, padding=0),
            nn.ReLU(inplace=True),              #output_size: H*W = 12*30*32
            # nn.MaxPool2d(kernel_size=5, stride=1)  #output_size: H*W = 10*28
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=1, padding=0),
            nn.ReLU(inplace=True),              #output_size: H*W = 8*26*64
            # nn.MaxPool2d(kernel_size=3, stride=1)  #output_size: H*W = 6*22
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8*26*64, 128),        
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
        return F.softmax(self.out(x), dim = 1)

class REINFORCE(object):
    def __init__(self, state_dim, action_dim, gamma, learning_rate, device) :
        super(REINFORCE,self).__init__()
        self.state_dim = state_dim         
        self.action_dim = action_dim
        self.gamma = gamma                                #Target Q计算参数
        self.learning_rate = learning_rate                 #学习率   
        self.count = 0               #计数器，记录网络更新次数
        self.device = device
        
        # 初始化policy网络
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(device)
        # 定义优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = learning_rate)  # 选取损失函数

    #  根据动作概率分布随机采样
    def take_action(self, state):
        state = torch.from_numpy(state).float()
        state = torch.unsqueeze(state, dim=0).to(self.device)   #增加一个维度用于输入CNN
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
        
    def learn(self, transition_dict):
        # 获得一回合的state、action、reward
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        reward_list = transition_dict['rewards']

        # 训练policy network
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):       #从最后一步算起
            state = torch.tensor([state_list[i]], dtype = torch.float32).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1,1).to(self.device)
            reward = reward_list[i]
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G   #每一步的损失函数
            loss.backward()   #反向计算梯度
        self.optimizer.step()   #梯度下降， 每一个回合

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
    net_paras = {"eval_net": agent.policy_net.state_dict(),"net_Adam": agent.optimizer.state_dict(), "epoch": ep}
    torch.save(obj=net_paras, f="CP0224/net_paras_save_Reinf.pth")

# 重载模型参数
def reload_parameters():
    #state_dict() 只保存网络参数，不保存网络结构
    reload_states = torch.load("CP0224/net_paras_save_Reinf.pth")#重载的reload_states 为一个字典
    agent.policy_net.load_state_dict(reload_states["eval_net"]) 
    agent.optimizer.load_state_dict(reload_states['net_Adam'])


if __name__ == '__main__':
    # hyper parameters
    NUM_CAR = 500       #车辆数
    NUM_NODE = 880    #站点数
    H_Grid = 22              #地理网格高度
    W_Grid = 40             #地理网格宽度
    MAX_EPISODES = 50           #运行次数（天数）
    MAX_EP_STEPS = 500    #每个episode的步数，时间间隔初设为1分钟，请求发出持续到1440分钟，即1440step   
    lr = 0.001
    gamma = 0.95              #折扣因子，越大->未来奖励影响  合理的小->敢于冒险
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #设置环境：输入为目标调度函数的权重系数，输出为奖励、下一时刻的状态、是否结束（？）
    ll = ctypes.cdll.LoadLibrary
    env = ll(r"/home/chwei/JeanLu/SharedTaxi/C_python/CP0224/libPolicy2_0224.so")  

    # create the arrays for the input of C's functions
    Car_Distribution = np.empty((NUM_NODE), dtype = float)  
    RU_Distribution_Past = np.empty((NUM_NODE), dtype = float)   
    RU_Distribution = np.empty((NUM_NODE), dtype = float)  
    RO_Distribution = np.empty((NUM_NODE), dtype = float)  
    Reward = np.zeros(1, dtype = float)

    # convert the numpy arrays or variables to the data type in C
    p2c = ConvertP2C()
    Car_Distribution = p2c.ConvertListFlt(Car_Distribution)
    RU_Distribution_Past = p2c.ConvertListFlt(RU_Distribution_Past)
    RU_Distribution = p2c.ConvertListFlt(RU_Distribution)
    RO_Distribution = p2c.ConvertListFlt(RO_Distribution)
    Reward = p2c.ConvertListFlt(Reward)

    #【状态】地理网格图——请求分布、车辆分布；
    s_dim =  4    
    #【动作】一维数组：调度函数中，乘客数分别为0，1，2，3时的附加值
    a_dim = 3       
    # 设置agent
    agent = REINFORCE(state_dim=s_dim,
                action_dim=a_dim,
                gamma = gamma,
                learning_rate = lr,
                device = device)

    # # 如需继续训练，重载参数
    # reload_parameters()

    # 记录开始时间
    t1 = time.time()
    # 确定曼哈顿区域内站点ID
    env.nodes_in_manhattan_fixed();

    # 存储用于观测ep_reward和每个状态的最大Q值（per step）
    return_list = []
    #开始训练，记忆池满后开始参数更新
    for ep in range(MAX_EPISODES):
        # 计算每个episode运行时间
        t_ep = time.time()
        #initialize all states in the beginning of the episode, including states of request, car and network
        env.sys_reset(Car_Distribution, RU_Distribution_Past, RU_Distribution, RO_Distribution)
        #初始化episode的累计奖励
        ep_reward = 0       
        #初始化存储s、a、ns、r的字典
        transition_dict = {'states': [], 'actions': [], 'rewards': [], 'next_states': []}              
        #遍历每一个时间步
        for t in range(MAX_EP_STEPS):     
            # 初始化二维状态数组
            CarDistribution_2D = np.zeros((H_Grid, W_Grid), dtype = float)
            RUDistribution_Past_2D = np.zeros((H_Grid, W_Grid), dtype = float)
            RUDistribution_2D = np.zeros((H_Grid, W_Grid), dtype = float)
            RODistribution_2D = np.zeros((H_Grid, W_Grid), dtype = float)
            rows = -1
            #将Car_Distribution一维状态【880，】表示转为二维【22，40】
            for i in range(NUM_NODE):                
                if (i % 40 == 0):
                    rows += 1
                CarDistribution_2D[rows][i % 40] = np.array(Car_Distribution)[i]
                RUDistribution_Past_2D[rows][i % 40] = np.array(RU_Distribution_Past)[i]
                RUDistribution_2D[rows][i % 40] = np.array(RU_Distribution)[i]
                RODistribution_2D[rows][i % 40] = np.array(RO_Distribution)[i]
            #获取全局共享状态：路网中车辆分布和请求分布
            s_global = np.stack((CarDistribution_2D, RUDistribution_Past_2D, RUDistribution_2D, RODistribution_2D), axis = 0)   #(3, 22, 40)       
            #获取动作值，并转换数据类型为float
            a = agent.take_action(s_global)        #返回的a为numpy.int
            #将动作丢入环境获取奖励和下一时刻全局状态
            t_c = p2c.ConvertInt(t)
            a_c = p2c.ConvertInt(a)
            env.execution(a_c, t_c, Car_Distribution, RU_Distribution_Past, RU_Distribution, RO_Distribution, Reward)
            r = np.array(Reward)[0] 

            #t+1时刻状态，将一维状态表示转为二维
            CarDistribution_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)
            RUDistribution_Past_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)
            RUDistribution_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)
            RODistribution_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)

            rows = -1
            for i in range(NUM_NODE):                
                if (i % 40 == 0):
                    rows += 1
                CarDistribution_2D_[rows][i % 40] = np.array(Car_Distribution)[i]
                RUDistribution_Past_2D_[rows][i % 40] = np.array(RU_Distribution_Past)[i]
                RUDistribution_2D_[rows][i % 40] = np.array(RU_Distribution)[i]
                RODistribution_2D_[rows][i % 40] = np.array(RO_Distribution)[i]
            #获取t+1全局共享状态：路网中车辆分布和请求分布
            s_global_ = np.stack((CarDistribution_2D_, RUDistribution_Past_2D_, RUDistribution_2D_, RODistribution_2D_), axis = 0)   #(3,22,40)       
            # print("time id:", t)
            # print(CarDistribution_2D_)
            #存储数据 在字典
            transition_dict['states'].append(s_global)
            transition_dict['actions'].append(a)
            transition_dict['rewards'].append(r)
            transition_dict['next_states'].append(s_global_)

            #更新状态
            s_global = s_global_
            #更新累计奖励
            ep_reward += r 
            #随机打印一些动作值
            if t==0:
                print("action in t=: ",a)  
            elif t==10:
                print("action in t=: ",a)  
            elif t==200:
                print("action in t=: ",a)  
            elif t==300:
                print("action in t=: ",a)  
            elif t==360:
                print("action in t=: ",a)  
            elif t==390:
                print("action in t=: ",a)  
            elif t==421:
                print("action in t=: ",a)  
            elif t==488:
                print("action in t=: ",a)  
            elif t==489:
                print("action in t=: ",a)  
            else:
                pass
            #如果达到了最大步数（一天结束），打印奖励和噪声大小，并退出循环
            if t == MAX_EP_STEPS - 1:
                print('Episode:', ep, ' Reward: %i' % int(ep_reward))
                print('Running time: ', time.time() - t_ep) 
                break
        return_list.append(ep_reward)
        agent.learn(transition_dict)               
    print('Running time: ', time.time() - t1) 
    
    # 存储参数，方便继续训练
    save_parameters()

    # 可视化每个episode的Reward
    save_date = '0315'   #今天是几月几号呢:)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on env')
    plt.savefig('CP0224/return_REINFORCE{}.png'.format(save_date))
    plt.show()


