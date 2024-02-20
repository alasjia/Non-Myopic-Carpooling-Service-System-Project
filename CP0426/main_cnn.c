#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "env3_cnn.h"

/*
Created on 26th Apirl 2023
@author: lyj

经过0401版本一番波折，又改回之前状态不连续的版本（0401状态连续版本太容易出错）

main(), initial_net(), execution(),  car_state_generate()

gcc 编译生成动态库(lib_XXX.so):
gcc -o lib_400veh_r1.so -shared -fPIC main_cnn.c
*/

int main()
{
    // srand((unsigned)time(NULL)); // 随机数种子随时间变化
    /*Output for DDPG*/
    // 全局共享状态与车辆独立状态
    float Car_Distribution[NUMNODE];        // 未来15min时可用车辆分布 - 当前可用车辆分布
    float Req_UpDistribution_past[NUMNODE]; // 过去15mins内所有请求的分布
    float Req_Up_Distribution[NUMNODE];     // 新请求上车分布
    float Req_Off_Distribution[NUMNODE];    // 新请求下车分布
    // 单个步长（时间间隔）返回的实际奖励
    float Reward = 0; // 改为浮点数04.18

    clock_t start_all, finish_all; // 用于统计程序运行时间
    float Total_time;
    int t_interval;
    int i;
    int Alpha;
    float ep_reward = 0;
    int ep = 0;
    clock_t start_ep, finish_ep;

    // start timing
    start_all = clock();
    // 初始化路网信息
    initial_net(Car_Distribution, Req_UpDistribution_past, Req_Up_Distribution, Req_Off_Distribution);
    // input_check();
    // 设置派车策略：0，1，2
    Alpha = 0;
    // begin execution
    for (ep = 0; ep < 2; ep++)
    {
        // 初始化：系统收益+车辆状态中时间
        initial_ep();
        // ep计时
        start_ep = clock();
        // 开始一天的遍历
        for (t_interval = 0; t_interval < 1440; t_interval++)
        {
            // Alpha = rand() % 3; // 随机选择
            Reward = execution(Alpha, t_interval, Car_Distribution, Req_UpDistribution_past, Req_Up_Distribution, Req_Off_Distribution);
            // // check
            // printf("--------------------time%d----------------------\n", t_interval);
            // for (i = 0; i < NUMNODE; i++)
            // {
            //     if (Car_Distribution[i] != 0)
            //         printf("node%d: %f\n", i, Car_Distribution[i]);
            //     if (Req_Off_Distribution[i] != 0)
            //     printf("node%d: %f\n", i, Req_Off_Distribution[i]);
            // }
            ep_reward += Reward;
            // printf("step%d reward: %f \n", t_interval, Reward);
        }
        finish_ep = clock();
        printf("ep%d reward: %f    req_num: %d\n", ep, ep_reward, ReqNum_Total);
        printf("using time: %f seconds\n", (float)(finish_ep - start_ep) / CLOCKS_PER_SEC); // 输出时间
        ep_reward = 0;
    }

    // output the results
    output();
    // end timing
    finish_all = clock();
    Total_time = (float)(finish_all - start_all) / CLOCKS_PER_SEC; // 单位换算成秒
    printf("Total Time: %f seconds\n", Total_time);                // 输出时间

    return 0;
}

// reset the system for the beginning of every epoch
void initial_net(float Car_Distribution[NUMNODE], float Req_UpDistribution_past[NUMNODE], float Req_Up_Distribution[NUMNODE], float Req_Off_Distribution[NUMNODE])
{
    // 用于全局状态的初始化输出：车辆分布、请求分布
    int car;
    int req;
    int node;
    // 确定曼哈顿内站点
    nodes_in_manhattan_fixed();
    // 路网数据输入
    mincost_input();
    /*
    初始时刻：
    可用车辆分布图为全部车辆的分布；
    过去30min的请求分布是0；
    开始时刻无新请求，Req_Up_Distribution为0。
    */
    for (node = 0; node < NUMNODE; node++)
    {
        Car_Distribution[node] = 0;
        Req_UpDistribution_past[node] = 0;
        Req_Up_Distribution[node] = 0;
        Req_Off_Distribution[node] = 0;
    }
    for (car = 0; car < NUMCAR; car++)
        Car_Distribution[CarState_NodeID[car]] += 1;
}

/*
execute the system for 1 min(step):
input: current time(min),    current state
output: vehicle distribution after the step, request distribution of next step, reward of the step
*/
float execution(int Action, int interval_id, float Car_Distribution[NUMNODE], float Req_UpDistribution_past[NUMNODE], float Req_Up_Distribution[NUMNODE], float Req_Off_Distribution[NUMNODE])
{
    float reward; // 最终输出的每个时间间隔内的系统收益（奖励）

    int requestid = -1; // initialize the requestid to -1
    int carid;
    int uptaskid, offtaskid;
    int i, j;
    int car, req, node;
    int cars_dis_sort[MAX_MATCH_NUM];                     // 存储按上车点距离升序后的车辆id
    int cars_psg_sort[MAX_MATCH_NUM];                     // 存储按空座数升序后的车辆id
    int cars_vac[MAX_MATCH_NUM], cars_occ[MAX_MATCH_NUM]; // 空车集合和有客车辆集合
    int num_vac, num_occ;                                 // 空车集合和有客车辆集合中数量

    ReqNum_Intvl = 0;         // 指定时间间隔内请求数量清零
    for (j = 0; j < 500; j++) // 请求存储数组清零
        Exec_Reqids[j] = 0;
    // 记录当前时间间隔内的请求id和请求数量
    for (j = ReqId_Last; j < ReqId_Last + MAX_REQ_INTERVAL; j++)
    {
        if ((ReqList_ReqTime[j] >= interval_id) && (ReqList_ReqTime[j] < (interval_id + 1))) // if (ReqList_ReqTime[j] == interval_id )
        {
            Exec_Reqids[ReqNum_Intvl] = j;
            ReqNum_Intvl += 1;
        }
    }
    ReqId_Last = ReqId_Last + ReqNum_Intvl; // 更新请求的累积进度

    // 根据delta t内的请求进行更新、匹配
    // 如果delta t 内有请求，则action进行正常选择
    for (j = 0; j < ReqNum_Intvl; j++)
    {
        // printf("---------------------------第%d个请求----------------------------\n", j);
        Min_Dispatch_Cost = 1000000000; // 最小损失时间变量，初始设为最大
        BestCar = -1;                   // 初始最佳车辆ID为-1
        requestid = Exec_Reqids[j];
        uptaskid = ReqList_UpTaskId[requestid]; // 获取新请求的上下车任务id
        offtaskid = ReqList_OffTaskId[requestid];
        // 对于每个请求，需要初始化TaskChain_Candi为-1
        for (i = 0; i < 50; i++)
            TaskChain_Candi[i] = -1;

        // 遍历所有车辆，更新状态至新请求发出时刻
        for (i = 0; i < NUMCAR; i++)
        {
            update(requestid, i);
        }

        /*policy1: 无差别地寻找最小损失的 车辆*/
        if (Action == 0)
        {
            // 初始化cars_dis_sor为-1
            for (i = 0; i < MAX_MATCH_NUM; i++)
                cars_dis_sort[i] = -1;
            // 搜索附近车辆数并按照距离升序排序【附近车辆】
            insert_sort_dis(cars_dis_sort, requestid);
            // for (i = 0; i < MAX_MATCH_NUM; i++)
            //     printf("%d ", cars_dis_sort[i]);
            // printf("\n");

            // 请求匹配：对搜索获得的最近车辆按距离从近到远进行遍历
            for (i = 0; i < MAX_MATCH_NUM; i++)
            {
                if (cars_dis_sort[i] < 0) // 如果读取到-1即 结束
                    break;
                carid = cars_dis_sort[i];
                // printf("carid%d: %d\n", i, carid);
                if (CarState_ChairNum[carid] > 3 || CarState_ChairNum[carid] < 0)
                    printf("Policy1: Awfully wrong!!!!\n");
                match(requestid, carid);
                // 如果等待时间和绕行时间均为0，那么说明该匹配方案已经最优，无需遍历其他车辆
                if (Min_Dispatch_Cost == 0)
                    break;
            }
        }

        /*policy2: 优先派遣有客车辆（中损失最小的车辆），在无符合条件有客车辆后选择空车中最优*/
        else if (Action == 1)
        {
            // 空车集合和有客车辆集合及数量，先清零
            num_vac = 0;
            num_occ = 0;
            for (i = 0; i < MAX_MATCH_NUM; i++)
            {
                cars_occ[i] = -1;
                cars_vac[i] = -1;
            }

            //  搜索附近车辆，并获取空车id集合和有客车辆id集合
            vacant_and_occupied(cars_vac, &num_vac, cars_occ, &num_occ, requestid);
            // printf("req%d：\n", requestid);
            // printf("空车数量：%d\n", num_vac);
            // printf("随机抽查：%d\n", CarState_ChairNum[cars_vac[1]]);
            // printf("有客车数量：%d\n", num_occ);
            // printf("随机抽查：%d\n", CarState_ChairNum[cars_occ[5]]);
            // 请求匹配：先遍历有客车辆，从中择优
            for (i = 0; i < num_occ; i++)
            {
                carid = cars_occ[i];
                if (CarState_ChairNum[carid] > 3 || CarState_ChairNum[carid] < 0)
                {
                    printf("有客车：Awfully wrong!!!!\n");
                    printf("问题出在   req: %d,  car: %d\n", requestid, carid);
                    // exit(0);
                }

                match(requestid, carid);

                // 如果等待时间和绕行时间均为0，那么说明该匹配方案已经最优，无需遍历其他车辆
                if (Min_Dispatch_Cost == 0)
                    break;
            }

            // 如果有客车辆集合中无符合约束条件的车辆（情况应该极少），从空车集合中择优
            if (BestCar < 0)
            {
                for (i = 0; i < num_vac; i++)
                {
                    carid = cars_vac[i];
                    if (CarState_ChairNum[carid] > 3 || CarState_ChairNum[carid] < 0)
                        printf("空车：Awfully wrong!!!!\n");
                    match(requestid, carid);
                    // 如果等待时间和绕行时间均为0，那么说明该匹配方案已经最优，无需遍历其他车辆
                    if (Min_Dispatch_Cost == 0)
                        break;
                }
            }
        }

        /*policy3: 激进派遣有客车辆，按照3-2-1-0顺序搜索，寻找第一个满足约束条件的车辆*/
        else if (Action == 2)
        {
            // 初始化cars_dis_sor为-1
            for (i = 0; i < MAX_MATCH_NUM; i++)
                cars_psg_sort[i] = -1;
            //  先找出最近的车，再对所有可用车辆按照空座数升序排序
            insert_sort_dis(cars_psg_sort, requestid);
            insert_sort_psg(cars_psg_sort, requestid);
            // for (i = 0; i < MAX_MATCH_NUM; i++)
            //     printf("%d ", cars_psg_sort[i]);
            // printf("\n");
            // 请求匹配：按照座位数3-2-1-0的顺序遍历，选择满足约束条件的某座位数集合中损失最小车辆
            for (i = 0; i < MAX_MATCH_NUM; i++)
            {
                if (cars_psg_sort[i] < 0) // 如果读取到-1即结束
                    break;
                carid = cars_psg_sort[i];
                // printf("carid%d: %d\n", i, carid);
                if (CarState_ChairNum[carid] > 3 || CarState_ChairNum[carid] < 0)
                    printf("Policy3: Awfully wrong!!!!\n");
                match(requestid, carid);     // 执行匹配
                if ((i + 1) < MAX_MATCH_NUM) // 防止溢出
                {
                    if (CarState_ChairNum[carid] == 3 && CarState_ChairNum[cars_psg_sort[i + 1]] == 2)
                    {                     // cars_psg_sort中next car为2人车，说明3人车遍历完毕，以下同理
                        if (BestCar > -1) // 如果有可用车辆，则跳出循环
                            break;
                    }
                    else if (CarState_ChairNum[carid] == 2 && CarState_ChairNum[cars_psg_sort[i + 1]] == 1)
                    {
                        if (BestCar > -1)
                            break;
                    }
                    else if (CarState_ChairNum[carid] == 1 && CarState_ChairNum[cars_psg_sort[i + 1]] == 0)
                    {
                        if (BestCar > -1)
                            break;
                    }
                    else
                        ; // do nothing
                }
            }
        }
        else
            printf("policy setting is wrong!!!\n");

        // 当所有车辆都遍历完毕，当该请求成功匹配到车辆的情况下，在将candi赋值给最终数组进行更新
        if (BestCar > -1)
        { // 仅当匹配到车辆时更新
            // Car_RouteList[BestCar][uptaskid] = 1; // 将匹配成功的车辆与任务记录到任务清单中
            // Car_RouteList[BestCar][offtaskid] = 1;
            CarState_FirstTask[BestCar] = TaskChain_Candi[0]; // 更新匹配到新请求的车辆的当前任务（仅当没有任务时才会变化）

            // // check the best car's task chain every step
            // printf("BestCar %d: ", BestCar);
            // for (i = 0; i < 10; i++)
            // 	printf("--> %d ", TaskChain_Candi[i]);
            // printf("\n");
            if (TaskChain_Candi[1] < 0) // 匹配成功后至少有两个任务
                printf("task chain is wrong!!!\n");

            for (i = 1; TaskChain_Candi[i] > -1; i++)
            { // 更新NextTask任务链列表
                TaskList_NextTaskID[TaskChain_Candi[i - 1]] = TaskChain_Candi[i];
            }
        }
    }

    // 如果delta t 内无请求，则action为空集合???
    //  else

    // 函数返回：t+1时刻状态+ 实际奖励
    // 状态数组清零
    for (node = 0; node < NUMNODE; node++)
    {
        Car_Distribution[node] = 0;
        Req_UpDistribution_past[node] = 0;
        Req_Up_Distribution[node] = 0;
        Req_Off_Distribution[node] = 0;
    }

    // a. 当前可用车辆数分布
    // ps:可用车辆：当前车上人数小于满载数
    for (car = 0; car < NUMCAR; car++)
    {
        if (CarState_ChairNum[car] < NUMCHAIR)
            Car_Distribution[CarState_NodeID[car]] += 1;
    }
    // b. 请求在过去30min内的地理分布
    req = ReqId_Last - 1; // 当前时刻最后一个请求的ID应该是累积数量-1 衔接处???  相当于不准确
    while (ReqList_ReqTime[req] <= interval_id)
    {
        if (ReqList_ReqTime[req] > (interval_id - 30))
            Req_UpDistribution_past[ReqList_UpNode[req]] += 1;
        req--;
        if (req < 0)
            break;
    }
    // c.新请求上车、下车位置分布
    for (j = 0; j < ReqNum_Intvl; j++)
    {
        req = Exec_Reqids[j];
        Req_Up_Distribution[ReqList_UpNode[req]] += 1;
        Req_Off_Distribution[ReqList_OffNode[req]] -= 1;
        if (ReqList_UpNode[req] == ReqList_OffNode[req])
            printf("pickup node == dropoff node!");
    }
    // n. 当前时间间隔内的实际奖励
    reward = 0; // 在输入环境前将1个时间步的累计奖励清零
    for (j = 0; j < ReqNum_Intvl; j++)
    {
        req = Exec_Reqids[j];
        // reward += SysIncreRevd[req] - SysIncreCost[req]; // 绕行、等待不扣钱
        reward += SysIncreRevd[req] - SysIncreCost[req] - SysIncreDetr[req] - SysIncreWait[req];
        // printf("interval%d's reward: %3f\n", ReqList_ReqTime[req], reward[0]);
        // printf("check SysIncreCost[%d]: %.3f\n", req, SysIncreCost[req]);
    }

    // // show the distribution of car in the end of every interval
    // if (interval_id < 50)
    //     printf("time%d:\n", interval_id);
    // for (i = 0; i < NUMNODE; i++)
    // {
    //     if (Req_UpDistribution_past[i] > 0 && interval_id < 50)
    //         printf("%.0f    ", Req_UpDistribution_past[i]);
    //     if (Req_UpDistribution_past[i] < 0 || Car_Distribution[i] < 0 || Req_Up_Distribution[i] < 0)
    //         printf("OUTPUT VALUE < 0!!!\n");
    //     else if (Req_Off_Distribution[i] > 0)
    //         printf("Req_Off_Distribution > 0!!!\n");
    // }
    // if (interval_id < 50)
    //     printf("\n");
    return reward;
}
