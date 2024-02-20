#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "env3_cnn.h"

/*
Created on Saturday Feb 18 2023
@author: lyj

简化状态空间为3：
1、请求在过去15min 内的地理分布
2、以当前时间间隔内到达新请求上车位置为中心，附近座位数的地理分布
3、可用车辆数未来15min时的分布-当前分布
修改动作策略为离散型：
Action1：无差别地寻找最小损失的车辆
Action2：将车辆分为两种类别，有客车辆和空车，寻找有客车辆中最小损失，在无符合条件有客车辆后选择空车中最优
Action3：激进策略，按照3-2-1-0顺序搜索，寻找第一个满足约束条件的车辆
Custom functions：
main(), sys_reset(), execution(),  car_state_generate()

gcc 编译生成动态库(lib_XXX.so):
gcc -o libPolicy2_0224.so -shared -fPIC main_cnn.c
*/

int main()
{
    /*Output for DDPG*/
    // 全局共享状态与车辆独立状态
    float CarChange_Distribution[NUMNODE]; // 未来15min时可用车辆分布 - 当前可用车辆分布
    float Req_Distribution_15[NUMNODE];    // 过去15mins内所有请求的分布
    float Seat_Distribution_Req[NUMNODE];  // 以新请求上车点为中心，附近空座数分布
    // 单个步长（时间间隔）返回的实际奖励
    float Reward[1];

    clock_t start_all, finish_all; // 用于统计程序运行时间
    float Total_time;
    int t_interval;
    int i;
    float Alpha;

    // start timing
    start_all = clock();
    // reset system
    nodes_in_manhattan_fixed();
    sys_reset(CarChange_Distribution, Req_Distribution_15, Seat_Distribution_Req);
    // input_check();
    // 设置派车策略：0，1，2
    Alpha = 1;
    // begin execution
    for (t_interval = 0; t_interval < 1440; t_interval++)
    {
        execution(Alpha, t_interval, CarChange_Distribution, Req_Distribution_15, Seat_Distribution_Req, Reward);
        // // check
        // printf("--------------------time%d----------------------\n", t_interval);
        // for (i = 0; i < NUMNODE; i++)
        // {
        //     if (Car_Distribution_15[i] != 0)
        //         printf("node%d: %f\n", i, Car_Distribution_15[i]);
        // }
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
void sys_reset(float CarChange_Distribution[NUMNODE], float Req_Distribution_15[NUMNODE], float Seat_Distribution_Req[NUMNODE])
{
    // 用于全局状态的初始化输出：车辆分布、请求分布
    int car;
    int req;
    int node;
    // 请求数据输入
    reqdata_input();
    // 路网数据输入
    mincost_input();
    // 初始化各个数组
    init_arr(CarChange_Distribution, Req_Distribution_15, Seat_Distribution_Req);
    // 请求与任务转换
    trans_Reqid_Taskid();
    /*
    初始化可用车辆分布变化图  初始时刻默认无变化为0
    过去15min的请求分布是0，
    新请求附近空座，开始时刻无新请求，故为0
    */
}

/*
execute the system for 1 min(step):
input: current time(min),    current state
output: vehicle distribution after the step, request distribution of next step, reward of the step
*/
void execution(int Action, int interval_id, float CarChange_Distribution[NUMNODE], float Req_Distribution_15[NUMNODE], float Seat_Distribution_Req[NUMNODE], float reward[1])
{
    int requestid = -1; // initialize the requestid to -1
    int carid;
    int uptaskid, offtaskid;
    int i, j;
    int car, req, node;
    float action_car;
    int cars_dis_sort[MAX_MATCH_NUM];                     // 存储按上车点距离升序后的车辆id
    int cars_psg_sort[MAX_MATCH_NUM];                     // 存储按空座数升序后的车辆id
    int cars_vac[MAX_MATCH_NUM], cars_occ[MAX_MATCH_NUM]; // 空车集合和有客车辆集合
    int num_vac, num_occ;                                 // 空车集合和有客车辆集合中数量

    // 找出一分钟内对应的请求id    可以在python中完成
    int req_num = 0;      // 存储时间间隔内请求数量
    int exec_reqids[500]; // 存储指定时间间隔内的请求id

    int aa;

    // 记录当前时间间隔内的请求id和请求数量
    for (j = 0; j < NUMREQ_RUN; j++)
    {
        if ((ReqList_ReqTime[j] >= interval_id) && (ReqList_ReqTime[j] < (interval_id + 1))) // if (ReqList_ReqTime[j] == interval_id )
        {
            exec_reqids[req_num] = j;
            req_num += 1;
        }
    }

    // 根据delta t内的请求进行更新、匹配
    // 如果delta t 内有请求，则action进行正常选择
    for (j = 0; j < req_num; j++)
    {
        // printf("---------------------------第%d个请求----------------------------\n", j);
        Min_Dispatch_Cost = 1000000000; // 最小损失时间变量，初始设为最大
        BestCar = -1;                   // 初始最佳车辆ID为-1
        requestid = exec_reqids[j];
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
                if (cars_dis_sort[i] < 0) // 如果读取到-1即结束
                    break;
                carid = cars_dis_sort[i];
                // printf("carid%d: %d\n", i, carid);
                if (CarState_ChairNum[carid] > 3 || CarState_ChairNum[carid] < 0)
                    printf("Awfully wrong!!!!\n");
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
                    printf("有客车：Awfully wrong!!!!\n");
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
            //  对所有可用车辆按照空座数升序排序(相同座位数按照距离升序)
            insert_sort_dis(cars_psg_sort, requestid);
            insert_sort_psg(cars_psg_sort, requestid);
            // for (i = 0; i < MAX_MATCH_NUM; i++)
            //     printf("%d ", cars_psg_sort[i]);
            // printf("\n");
            // 请求匹配：对搜索获得的最近车辆按距离从近到远进行遍历
            for (i = 0; i < MAX_MATCH_NUM; i++)
            {
                if (cars_psg_sort[i] < 0) // 如果读取到-1即结束
                    break;
                carid = cars_psg_sort[i];
                // printf("carid%d: %d\n", i, carid);
                if (CarState_ChairNum[carid] > 3 || CarState_ChairNum[carid] < 0)
                    printf("Awfully wrong!!!!\n");
                match(requestid, carid);
                // 如果有满足约束条件的车辆，直接采用（相当于采用最近的2人车辆）
                if (BestCar > -1)
                    break;
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

            for (i = 1; TaskChain_Candi[i] != -1; i++)
            { // 更新NextTask任务链列表
                TaskList_NextTaskID[TaskChain_Candi[i - 1]] = TaskChain_Candi[i];
            }
        }
    }

    // 如果delta t 内无请求，则action为空集合???
    //  else

    // 函数返回：全局共享状态的车辆分布和请求分布 + 实际奖励
    // 状态数组清零
    for (node = 0; node < NUMNODE; node++)
    {
        Car_Distribution[node] = 0;
        Car_Distribution_15[node] = 0;
        Req_Distribution_15[node] = 0;
        CarChange_Distribution[node] = 0;
        Avai_Seats[node] = 0;
        Seat_Distribution_Req[node] = 0;
    }

    // a. 车辆数未来15min时的分布-当前分布
    // 遍历所有车辆，更新位置状态至15mins后
    for (car = 0; car < NUMCAR; car++)
        update_15(interval_id, i);
    // 存储两时刻分布           ps:可用车辆：当前车上人数小于满载数
    for (car = 0; car < NUMCAR; car++)
    {
        if (CarState_ChairNum[car] < NUMCHAIR)
            Car_Distribution[CarState_NodeID[car]] += 1;
        if (CarState_ChairNum_15[car] < NUMCHAIR)
            Car_Distribution_15[CarState_Node_15[car]] += 1;
    }
    // 计算差值
    for (node = 0; node < NUMNODE; node++)
        CarChange_Distribution[node] = Car_Distribution_15[node] - Car_Distribution[node];

    // b. 请求在过去15min 内的地理分布
    req = 0;
    while (ReqList_ReqTime[req] <= interval_id)
    {
        if (ReqList_ReqTime[req] > (interval_id - 15))
            Req_Distribution_15[ReqList_UpNode[req]] += 1;
        req++;
        if (req >= NUMREQ)
            break;
    }

    // c.以当前时间间隔内到达新请求上车位置为中心，附近座位数的地理分布
    // 先统计出当前时刻空座数分布
    for (car = 0; car < NUMCAR; car++)
    {
        node = CarState_NodeID[car];
        Avai_Seats[node] += (NUMCHAIR - CarState_ChairNum[car]);
    }
    // 再确定新请求上车位置，将其附近站点空座数复制
    for (j = 0; j < req_num; j++)
    {
        req = exec_reqids[j];
        // 搜索附近站点
        for (node = 0; node < NUMNODE; node++)
        {
            if (MinCost[node][ReqList_UpNode[req]] < SEARCH_DIS)
                Seat_Distribution_Req[node] = Avai_Seats[node];
        }
    }

    // n. 当前时间间隔内的实际奖励
    reward[0] = 0; // 在输入环境前将1个时间步的累计奖励清零
    for (j = 0; j < req_num; j++)
    {
        req = exec_reqids[j];
        reward[0] += SysIncreRevd[req] - SysIncreCost[req] - SysIncreDetr[req] - SysIncreWait[req];
        // printf("interval%d's reward: %3f\n", ReqList_ReqTime[req], reward[0]);
        // printf("check SysIncreCost[%d]: %.3f\n", req, SysIncreCost[req]);
    }

    // // show the distribution of car in the end of every interval
    // printf("Cars' Distribution:\n");
    // for (i = 0; i < 25; i++)
    // {
    // 	if (Car_Distribution[i] > 0)
    // 	{
    // 		for (car = 0; car < NUMCAR; car++)
    // 		{
    // 			if (CarState_NodeID[car] == i)
    // 				printf(" %d    ", car);
    // 		}
    // 	}
    // 	else
    // 		printf("%d    ", Car_Distribution[i] - 1);
    // 	if (i % 5 == 4)
    // 		printf("\n");
    // }
}
