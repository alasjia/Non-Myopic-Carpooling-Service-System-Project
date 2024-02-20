'''
Created on Monday Mar 27 2023
@author: lyj

建立于0401基础上
Q：一天中剩余时间的折扣回报
简化燃油计价：统一计价，不考虑乘客数   (r2)
简化reward：仅考虑乘客支付和燃油成本   (r2)

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


#Dueling Network
class VAnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(VAnet, self).__init__()
        self.conv1 = nn.Sequential(      
            nn.Conv2d(in_channels=state_dim, out_channels=8, kernel_size=(5,5), stride=1, padding=0),
            nn.ReLU(inplace=True),             #output_size: H*W = 18*36*32
            # nn.MaxPool2d(kernel_size=3, stride=1)  #output_size: H*W = 18*36
        )
        self.conv2 = nn.Sequential(      
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5,5), stride=1, padding=0),
            nn.ReLU(inplace=True),             #output_size: H*W = 14*32*32
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), stride=1, padding=0),
            nn.ReLU(inplace=True),              #output_size: H*W = 10*28*32
        )
        self.fc1A = nn.Sequential(
            nn.Linear(10*28*32, 128),        
            nn.ReLU()
        )
        self.fc2A = nn.Sequential(
            nn.Linear(128, action_dim)
        )
        self.fc1V = nn.Sequential(
            nn.Linear(10*28*32, 128),        
            nn.ReLU()
        )
        self.fc2V = nn.Sequential(
            nn.Linear(128, 1)
        )
    def forward(self, x):  
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)      # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）/x=x.squeeze()减少一个维度
        A = self.fc1A(x)   #优势头
        A = self.fc2A(A)
        V = self.fc1V(x)   #状态价值头
        V = self.fc2V(V)
        Q = V + A - A.mean(1).view(-1, 1)  #mean(1)对各行求均值       此处加入时间信息？？？
        return Q

class DDQN(object):
    def __init__(self, state_dim, action_dim, gamma, learning_rate, target_update, device, batch_size) :
        super(DDQN,self).__init__()
        self.state_dim = state_dim         
        self.action_dim = action_dim
        self.gamma = gamma                                #Target Q计算参数
        self.learning_rate = learning_rate                 #学习率        
        self.target_update = target_update    #目标网络更新频率
        self.count = 0               #计数器，记录网络更新次数
        self.device = device
        self.batch_size = batch_size
        
        # 初始化估计网络
        self.q_net = VAnet(self.state_dim, self.action_dim).to(device)
        # 初始化目标网络，初始时目标网络与估计网络参数相同     
        self.q_net_target = VAnet(self.state_dim, self.action_dim).to(device)
        
        # 定义优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = learning_rate)  # 选取损失函数
        self.mse_loss = nn.MSELoss()

    #  epsilon-greedy行为策略
    def take_action(self, state, epsilon):
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim, dtype = "int64")
        else:
            state = torch.from_numpy(state).float()            #增加一个维度用于输入CNN
            state = torch.unsqueeze(state, dim=0).to(self.device)
            action = self.q_net(state).detach()    #不明白detach() 
            action = action.argmax().item() #选择最大值对应的序号【0、1、2】;如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式
        return action

    def max_q_value(self, state):
        state = torch.from_numpy(state).float()            #增加一个维度用于输入CNN
        state = torch.unsqueeze(state, dim=0).to(self.device)
        return self.q_net(state).max().item()  #在所有的q值中选择了一个最大的值，item()将一个元素从张量形式转为值（多个元素报错）
    
    def loss_value(self, s0, s1, a0, t0, r0):
        target = torch.zeros(1).to(self.device)
        #修改为输入CNN的格式
        s0 = torch.from_numpy(s0).float()            
        s0 = torch.unsqueeze(s0, dim=0).to(self.device)
        s1 = torch.from_numpy(s1).float()            
        s1 = torch.unsqueeze(s1, dim=0).to(self.device)
        r0 = torch.from_numpy(np.array(r0)).to(self.device)
        #计算t时刻的loss
        evalbl = self.q_net(s0)[0][a0]  #self.q_net(s0)输出(1,3)
        max_a1 = self.q_net(s1).argmax()  #argmax()获得最大值对应索引
        max_next_evalbl = self.q_net_target(s1)[0][max_a1]
        if(t0 == 1339):
            target = r0
        else:
            target = r0 + self.gamma * max_next_evalbl     #TD误差目标
        loss_t = self.mse_loss(evalbl, target)   
        return loss_t.item()

    def learn(self, transition_dict):
        # 从记忆库中采样bacth data
        state_batch = torch.tensor(transition_dict['states'], dtype = torch.float32).to(self.device)
        state_batch_ = torch.tensor(transition_dict['next_states'], dtype = torch.float32).to(self.device)
        action_batch = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        reward_batch = torch.tensor(transition_dict['rewards'], dtype = torch.float32).view(-1,1).to(self.device)
        # reward_batch = np.array(transition_dict['rewards'])
        timeid_batch = np.array(transition_dict['interval_ids'])#.view(-1,1)
        # 增加维度：view(-1,1)是torch的方法，reshape()是numpy的方法，前者可自动计算

        q_targets = torch.zeros([self.batch_size,1]).to(self.device)
        # 训练Q network
        q_values = self.q_net(state_batch).gather(dim = 1, index = action_batch)  #dim=1, index=[5,3] 相当于取第0行中第5个、第1行中第3个   type(q_values)结果是tensor
        #训练网络：【选取动作】  ；目标网络：【计算该动作价值】
        max_action = self.q_net(state_batch_).max(1)[1].view(-1,1)   #view(-1,1)将(batch,)变为(batch,1)   max(1)对每行进行操作后有values和indices两行值，所以[0]
        max_next_q_values = self.q_net_target(state_batch_).detach()     #其实不是很明白.detach()的意义
        max_next_q_values= max_next_q_values.gather(1, max_action)
        for i in range(self.batch_size):
            if(timeid_batch[i] == 1339):
                q_targets[i] = reward_batch[i]
            else:
                q_targets[i] = reward_batch[i] + self.gamma * max_next_q_values[i]     #TD误差目标
        # for i in range(self.batch_size):
        #     print(q_targets[i])
        agent_loss = self.mse_loss(q_values, q_targets)    #grad_fn=<MseLossBackward0>  为什么需要mean()？
        # print(torch.mean(agent_loss))  #grad_fn=<MeanBackward0>
        self.optimizer.zero_grad   #Pytorch中默认梯度会累积，这里需要显式将梯度设置为0
        agent_loss.backward()  #反向传播更新参数
        self.optimizer.step()

        #延迟更新目标网络
        if self.count % self.target_update == 0:
            self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.count += 1

class ReplayBuffer:
    '''经验回放池'''
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)    #队列，先进先出
    def add(self, state, action, reward, next_state, interval_id):  #将数据加入buffer
        self.buffer.append((state, action, reward, next_state, interval_id))
    def sample(self, batch_size):   #从buffer中采样数据，数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, interval_id = zip(*transitions)   #zip打包，zip(*)解压
        return np.array(state), action, reward, np.array(next_state), interval_id
    def size(self):
        return len(self.buffer)

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
    qnet_paras = {"q_eval_net": agent.q_net.state_dict(), "q_targ_net": agent.q_net_target.state_dict(), "qnet_Adam": agent.optimizer.state_dict(), "epoch": ep}
    torch.save(obj=qnet_paras, f="CP0426/qnet_paras_duel+double_400veh_r2.pth")

# 重载模型参数
def reload_parameters():
    #state_dict() 只保存网络参数，不保存网络结构
    reload_states = torch.load("CP0426/qnet_paras_duel+double_400veh_r2.pth")#重载的reload_states 为一个字典
    agent.q_net.load_state_dict(reload_states["q_eval_net"]) 
    agent.q_net_target.load_state_dict(reload_states['q_targ_net'])
    agent.optimizer.load_state_dict(reload_states['qnet_Adam'])

if __name__ == '__main__':
    # hyper parameters
    NUM_CAR = 400       #车辆数
    NUM_NODE = 880    #站点数
    H_Grid = 22              #地理网格高度
    W_Grid = 40             #地理网格宽度
    MAX_EPISODES = 100           #运行次数（天数）
    MAX_EP_STEPS = 1440    #每个episode的步数，时间间隔初设为1分钟，请求发出持续到1440分钟，即1440step   
    memory_capacity = 50000#100000   #记忆池容量，一般设为1e5-6，10000有点小
    minimal_size = 5000      #累积minimal_size条经验后开始训练，1000设置的太小
    lr = 0.001
    gamma = 0.99              #折扣因子，越大->未来奖励影响  合理的小->敢于冒险
    epsilo = 1        #初始随机率，1等于完全随机选择
    decay_rate = 0.9999      # 决定衰减周期
    target_update  = 200    #延迟更新频率
    batch_size = 32

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    #设置环境：输入为目标调度函数的权重系数，输出为奖励、下一时刻的状态
    ll = ctypes.cdll.LoadLibrary
    env = ll(r"/home/chwei/JeanLu/SharedTaxi/C_python/CP0426/lib_400veh_r2.so")  

    # create the arrays for the input of C's functions
    Car_Distribution = np.empty((NUM_NODE), dtype = float)  
    RU_Distribution_Past = np.empty((NUM_NODE), dtype = float)   
    RU_Distribution = np.empty((NUM_NODE), dtype = float)  
    RO_Distribution = np.empty((NUM_NODE), dtype = float)  

    # convert the numpy arrays or variables to the data type in C
    p2c = ConvertP2C()
    Car_Distribution = p2c.ConvertListFlt(Car_Distribution)
    RU_Distribution_Past = p2c.ConvertListFlt(RU_Distribution_Past)
    RU_Distribution = p2c.ConvertListFlt(RU_Distribution)
    RO_Distribution = p2c.ConvertListFlt(RO_Distribution)

    replay_buffer = ReplayBuffer(memory_capacity)
    #【状态】地理网格图——请求分布、车辆分布；
    s_dim =  5   #增加时间间隔id    
    #【动作】一维数组：调度函数中，乘客数分别为0，1，2，3时的附加值
    a_dim = 3       
    # 设置agent
    agent = DDQN(state_dim=s_dim,
                action_dim=a_dim,
                gamma = gamma,
                learning_rate = lr,
                target_update = target_update,
                device = device,
                batch_size = batch_size)

    # # 如需继续训练，重载参数
    # reload_parameters()

    # 记录开始时间
    t1 = time.time()
    # 系统变量初始化：确定曼哈顿内站点、路网数据输入、仿真系统内部数组、输出分布数组
    env.initial_net(Car_Distribution, RU_Distribution_Past, RU_Distribution, RO_Distribution)
    env.execution.restype = ctypes.c_float #设定好c函数返回值的类型，否则默认为int，输出值全为0
    # 初始化二维状态数组：t时刻和t+1时刻
    CarDistribution_2D = np.zeros((H_Grid, W_Grid), dtype = float)
    RUDistribution_Past_2D = np.zeros((H_Grid, W_Grid), dtype = float)
    RUDistribution_2D = np.zeros((H_Grid, W_Grid), dtype = float)
    RODistribution_2D = np.zeros((H_Grid, W_Grid), dtype = float)
    #t+1时刻状态，将一维状态表示转为二维
    CarDistribution_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)
    RUDistribution_Past_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)
    RUDistribution_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)
    RODistribution_2D_ = np.zeros((H_Grid, W_Grid), dtype = float)
    #将Car_Distribution一维状态【880，】表示转为二维【22，40】
    rows = -1
    for i in range(NUM_NODE):                
        if (i % 40 == 0):
            rows += 1
        CarDistribution_2D[rows][i % 40] = np.array(Car_Distribution)[i]
        RUDistribution_Past_2D[rows][i % 40] = np.array(RU_Distribution_Past)[i]
        RUDistribution_2D[rows][i % 40] = np.array(RU_Distribution)[i]
        RODistribution_2D[rows][i % 40] = np.array(RO_Distribution)[i]
    #初始时刻的全局共享状态：路网中车辆分布和请求分布
    Time_2D = np.zeros((H_Grid, W_Grid))
    s_global = np.stack((CarDistribution_2D, RUDistribution_Past_2D, RUDistribution_2D, RODistribution_2D, Time_2D), axis = 0)   #(3, 22, 40)       
        
    # 存储用于观测ep_reward和每个状态的最大Q值（per step）
    return_list = []
    loss_list = []
    max_q_value_list = []
    max_q_value = 0
    #初始化episode的累计奖励、总累积奖励和ep次数
    ep_reward = 0       
    total_ep_reward = 0
    
    # 开始训练
    for ep in range(MAX_EPISODES):
        env.initial_ep()           #初始化：系统收益+车辆状态中时间
        t_ep = time.time()        # 计算每个episode运行时间
        for t in range(MAX_EP_STEPS):     
            # check
            # if(t % 222 == 0):
                # print("--------------------step", t, "--------------------------")
                # for i in range(NUM_NODE):
                #     if(np.array(RU_Distribution)[i] != 0):
                #         print(i, ":", np.array(RU_Distribution)[i])
                # for i in range(NUM_NODE):
                #     if(np.array(RO_Distribution)[i] != 0):
                #         print(i, ":", np.array(RO_Distribution)[i])            
            
            #获取动作值，并转换数据类型为float
            a = agent.take_action(s_global, epsilo)        #返回的a为numpy.int

            # max_q_value = agent.max_q_value(s_global) * 0.005 + max_q_value *0.995 #平滑处理
            max_q_value = agent.max_q_value(s_global)             #最大状态价值函数
            max_q_value_list.append(max_q_value)

            #将动作丢入环境获取奖励和下一时刻全局状态
            t_c = p2c.ConvertInt(t)
            a_c = p2c.ConvertInt(a)
            #输出单个时间间隔内的奖励值，c函数直接输出python的float
            r_step = env.execution(a_c, t_c, Car_Distribution, RU_Distribution_Past, RU_Distribution, RO_Distribution)
            r_step = np.float32(r_step)   #转为numpy数据类型float32
            # print("step", t, "  reward: ", r_step)     #check 
            #此时各Distribution进入t+1时刻，所以更新Distribution_2D
            rows = -1
            for i in range(NUM_NODE):                
                if (i % 40 == 0):
                    rows += 1
                CarDistribution_2D_[rows][i % 40] = np.array(Car_Distribution)[i]
                RUDistribution_Past_2D_[rows][i % 40] = np.array(RU_Distribution_Past)[i]
                RUDistribution_2D_[rows][i % 40] = np.array(RU_Distribution)[i]
                RODistribution_2D_[rows][i % 40] = np.array(RO_Distribution)[i]
                Time_2D[rows][i % 40] = t+1
            #获取t+1全局共享状态：路网中车辆分布和请求分布
            s_global_ = np.stack((CarDistribution_2D_, RUDistribution_Past_2D_, RUDistribution_2D_, RODistribution_2D_, Time_2D), axis = 0)   #(3,22,40)       
            #存储数据 在记忆池
            replay_buffer.add(s_global, a, r_step, s_global_, t)

            if replay_buffer.size() > minimal_size:   #开启训练模式
                #探索力度衰减
                epsilo *= decay_rate    
                b_s, b_a, b_r, b_ns, b_intvl = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'rewards': b_r, 'next_states': b_ns, 'interval_ids': b_intvl}
                agent.learn(transition_dict)
            #-----------check-----------
            if t % 100 == 0:
                print("action in t", t, " =: ",a)  
                print(max_q_value)
            #-------------------------------
            #存储可视化信息：loss
            if(t %100 == 0):
                loss_list.append(agent.loss_value(s_global, s_global_, a, t, r_step))
            # 更新状态
            s_global = s_global_
            #更新累计奖励
            ep_reward += r_step 
            
        #如果一天结束，打印一天的累积奖励和用时
        total_ep_reward += ep_reward   #更新总奖励
        print('Episode:', ep, ' Reward: %i' % int(ep_reward))
        print('Total Reward: %i' % int(total_ep_reward))
        print('Running time: ', time.time() - t_ep) 
        return_list.append(ep_reward)    #每一天存储一次
        ep_reward = 0      #清零存储一天reward的变量
        
    print('Total Running time: ', time.time() - t1) 
    
    # 存储参数，方便继续训练
    save_parameters()

    #可视化每个episode的Reward
    save_date = '400veh_r2'   #今天是几月几号呢:)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN with Double+Dueling on env')
    plt.savefig('CP0426/return_DQN_Dueling+Double_{}.png'.format(save_date))
    plt.show()
    # 可视化每个时间间隔的loss
    interval_list = list(range(len(loss_list)))
    plt.plot(interval_list, loss_list)
    plt.xlabel('TimeId*100')
    plt.ylabel('Loss Values')
    plt.title('DQN with Double+Dueling on env')
    plt.savefig('CP0426/Loss_DQN_Dueling+Double_{}.png'.format(save_date))
    plt.show()
    #可视化每个状态的max_q_value
    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    # plt.axhline(?, c = 'orange', ls = '--')   #水平辅助线
    # plt.axhline(?, c = 'red', ls = '--')     
    plt.xlabel('Frames')
    plt.ylabel('Q values')
    plt.title('DQN with Double+Dueling on env')
    plt.savefig('CP0426/Qvalue_DQN_Dueling+Double_{}.png'.format(save_date))
    plt.show()




