'''
Created on Monday Mar 13 2023
@author: lyj

在DQN_CNN_old.py基础上改进：
优化replay buffer，将list改为dictionary；
增加可视化；
采用ActorCritic算法

Q：AC用Q学习还是SARSA（学习最优动作价值函数还是动作价值函数？）

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

#PolicyNet，输入某个状态，输出所有动作的概率
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

#ValueNet，输入某个状态，输出该状态的价值
class ValueNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ValueNet, self).__init__()
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
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)      # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）/x=x.squeeze()减少一个维度
        x = self.fc1(x)
        value = self.out(x)
        return value

class ActorCritic(object):
    def __init__(self, state_dim, action_dim, gamma, actor_lr, critic_lr, device) :
        super(ActorCritic,self).__init__()
        self.state_dim = state_dim         
        self.action_dim = action_dim
        self.gamma = gamma                                #Target Q计算参数   
        self.device = device
        #初始化存储s、a、r、ns、na的列表（未找到字典删除最远时刻数据的方法）
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.next_state_list = []
        self.next_action_list = []
        
        # 初始化policy网络和value网络
        self.actor = PolicyNet(self.state_dim, self.action_dim).to(device)
        self.critic = ValueNet(self.state_dim, self.action_dim).to(device)
        # 定义优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)  
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr) 

    #  根据动作概率分布随机采样
    def take_action(self, state):
        state = torch.from_numpy(state).float()
        state = torch.unsqueeze(state, dim=0).to(self.device)   #增加一个维度用于输入CNN
        probs = self.actor(state).detach()     #why？
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def max_q_value(self, state):
        state = torch.from_numpy(state).float()            
        state = torch.unsqueeze(state, dim=0).to(self.device)  #增加一个维度用于输入CNN
        return self.critic(state).item()

#注意！！！：此处learn()先依照网上A2C代码改写，应该是Q学习方法更新Critic
    def learn_single(self, s0, a0, r, s1, a1):
        #单步采样，为什么没有用到at+1？因为Critic就设置的不对！
        state = torch.tensor(s0, dtype = torch.float32).unsqueeze(dim = 0).to(self.device)
        next_state = torch.tensor(s1, dtype = torch.float32).unsqueeze(dim = 0).to(self.device)
        reward = torch.tensor(r, dtype = torch.float32).view(-1,1).to(self.device)
        action = torch.tensor(a0).view(-1,1).to(self.device)
        next_action = torch.tensor(a1).view(-1,1).to(self.device)
        #先更新价值网络：
        #TD目标
        td_target = reward + self.gamma * self.critic(next_state)
        #估计q值
        q_value = self.critic(state)
        #均方误差损失函数
        critic_loss = F.mse_loss(q_value, td_target.detach())
        #参数清零
        self.critic_optimizer.zero_grad()
        #计算网络梯度
        critic_loss.backward()
        #更新价值网络参数
        self.critic_optimizer.step()
        #再更新策略网络：
        #TDerror
        td_delta = td_target - q_value
        #计算action的损失
        log_probs = torch.log(self.actor(state).gather(1, action))
        actor_loss = -log_probs * td_delta.detach()
        #参数清零
        self.actor_optimizer.zero_grad()
        #计算网络梯度
        actor_loss.backward()  
        #更新策略网络参数
        self.actor_optimizer.step()

    def learn_multi(self, s0, a0, r, s1, a1, num_step):        
        #多步采样
        self.state_list.append(s0)
        self.next_state_list.append(s1)
        self.action_list.append(a0)
        self.next_action_list.append(a1)
        self.reward_list.append(r)
        if len(self.state_list) == num_step:   #若保存的数据可以进行n步更新     len()仅返回0维长度（shape第一个数）
            s1 = torch.tensor(s1, dtype = torch.float32).unsqueeze(dim = 0).to(self.device)  #将s转为torch格式用于输入NN
            temp_target = self.critic(s1)   #q(t+m)的估计值（tensor），temp_target.is_cuda返回True
            for i in reversed(range(num_step)):   #时间倒序，逐步向前加入真实reward，从而获得t时刻的TD目标值
                temp_target = self.gamma * temp_target + self.reward_list[i]
            td_target = temp_target        #td_target.is_cuda返回True
            #删除最前时刻的数据
            state = self.state_list.pop(0)   #t时刻state、action
            state = torch.tensor(state, dtype = torch.float32).unsqueeze(dim = 0).to(self.device) #2torch        state.shape()返回torch.Size([1, 4, 22, 40])
            action = self.action_list.pop(0)
            action = torch.tensor(action).view(-1,1).to(self.device)    #此处的view()是增加维度
            self.reward_list.pop(0)
            self.next_state_list.pop(0)
            self.next_action_list.pop(0)
            #先更新价值网络：
            q_value = self.critic(state)              
            critic_loss = F.mse_loss(q_value, td_target.detach()) #均方误差损失函数            
            self.critic_optimizer.zero_grad()  #参数清零            
            critic_loss.backward()  #计算网络梯度            
            self.critic_optimizer.step()  #更新价值网络参数
            #再更新策略网络：            
            td_delta = td_target - q_value #TDerror
            log_probs = torch.log(self.actor(state).gather(1, action)) #计算action的损失
            actor_loss = -log_probs * td_delta.detach()             #detach从计算图里摘出来，不参与参数更新
            self.actor_optimizer.zero_grad()  #参数清零            
            actor_loss.backward()    #计算网络梯度
            self.actor_optimizer.step() #更新策略网络参数


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
    policy_net_paras = {"eval_net": agent.actor.state_dict(),"net_Adam": agent.actor_optimizer.state_dict(), "epoch": ep}
    value_net_paras = {"eval_net": agent.critic.state_dict(),"net_Adam": agent.critic_optimizer.state_dict(), "epoch": ep}
    torch.save(obj=policy_net_paras, f="CP0224/policy_net_paras_save_AC.pth")
    torch.save(obj=value_net_paras, f="CP0224/value_net_paras_save_AC.pth")

# 重载模型参数
def reload_parameters():
    #state_dict() 只保存网络参数，不保存网络结构
    reload_states1 = torch.load("CP0224/policy_net_paras_save_AC.pth")#重载的reload_states 为一个字典
    agent.actor.load_state_dict(reload_states1["eval_net"]) 
    agent.actor_optimizer.load_state_dict(reload_states1['net_Adam'])
    #加载critic
    reload_states2 = torch.load("CP0224/value_net_paras_save_AC.pth")#重载的reload_states 为一个字典
    agent.critic.load_state_dict(reload_states2["eval_net"]) 
    agent.critic_optimizer.load_state_dict(reload_states2['net_Adam'])


if __name__ == '__main__':
    mm = 10              #TD采样步长
    # hyper parameters
    NUM_CAR = 500       #车辆数
    NUM_NODE = 880    #站点数
    H_Grid = 22              #地理网格高度
    W_Grid = 40             #地理网格宽度
    MAX_EPISODES = 50           #运行次数（天数）
    MAX_EP_STEPS = 500    #每个episode的步数，时间间隔初设为1分钟，请求发出持续到1440分钟，即1440step   
    batch_size = 32
    a_lr = 0.001
    c_lr = 0.01
    gamma = 0.99              #折扣因子，越大->未来奖励影响  合理的小->敢于冒险
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
    agent = ActorCritic(state_dim=s_dim,
                action_dim=a_dim,
                gamma = gamma,
                actor_lr = a_lr,
                critic_lr = c_lr,
                device = device)

    # # 如需继续训练，重载参数
    # reload_parameters()

    # 记录开始时间
    t1 = time.time()
    # 确定曼哈顿区域内站点ID
    env.nodes_in_manhattan_fixed();

    # 存储用于观测ep_reward和每个状态的最大Q值（per step）
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    #开始训练
    for ep in range(MAX_EPISODES):
        # 计算每个episode运行时间
        t_ep = time.time()
        #initialize all states in the beginning of the episode, including states of request, car and network
        env.sys_reset(Car_Distribution, RU_Distribution_Past, RU_Distribution, RO_Distribution)
        #初始化episode的累计奖励
        ep_reward = 0       
          
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
            #观测到当前状态St，根据当前策略做抽样
            a = agent.take_action(s_global)        #怎么3、4步之后概率就变得很极端了【0，0，1】？

            #记录价值网络最大Q值（三个动作对应的评估价值中最大的）
            max_q_value = agent.max_q_value(s_global) #无平滑处理
            max_q_value_list.append(max_q_value)

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
            #根据策略网络做决策，但a_不让智能体执行
            a_ = agent.take_action(s_global_)
            #m = 1 :单步TD目标（自举）; m > 1: 多步TD目标
            if mm == 1:
                agent.learn_single(s_global, a, r, s_global_, a_)
            else:
                agent.learn_multi(s_global, a, r, s_global_, a_, mm)
            #随机检查动作
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

            #更新状态
            s_global = s_global_
            #更新累计奖励
            ep_reward += r 
            #如果达到了最大步数（一天结束），打印奖励和噪声大小，并退出循环
            if t == MAX_EP_STEPS - 1:
                print('Episode:', ep, ' Reward: %i' % int(ep_reward))
                print('Running time: ', time.time() - t_ep) 
                break
        return_list.append(ep_reward)       
    print('Running time: ', time.time() - t1) 
    
    # # 存储参数，方便继续训练
    # save_parameters()

    # 可视化每个episode的Reward
    save_date = '0327'   #今天是几月几号呢:)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('ActorCritic_10step on env')
    plt.savefig('CP0224/return_ActorCritic_10step{}.png'.format(save_date))
    plt.show()
    #可视化每个状态的max_q_value
    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    # plt.axhline(?, c = 'orange', ls = '--')   #水平辅助线
    # plt.axhline(?, c = 'red', ls = '--')     
    plt.xlabel('Frames')
    plt.ylabel('Q values')
    plt.title('ActorCritic_10step on env')
    plt.savefig('CP0224/Qvalue_ActorCritic_10step{}.png'.format(save_date))
    plt.show()


