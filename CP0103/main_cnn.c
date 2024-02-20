#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "env3_cnn.h"

/*
Created on Wednesday December 14 2022
@author: lyj

The input data is the travel data of Manhattan yellow taxis.
Different from work in 1204, the action in this version include {aciton0, action1, action2, action3}, which represent  lost value with no passenger, 1 passenger, 2 passengers and 3 passengers in a taxi.
This version is suitable for full connection neural network.
Custom functions：
main(), sys_reset(), execution(),  car_state_generate()

gcc 编译生成动态库(lib_XXX.so):
gcc -o libPolicy1_cnn.so -shared -fPIC main_cnn.c
*/

int main()
{
    // clock_t start_all, finish_all; // 用于统计程序运行时间
    // float Total_time;
    // int t_interval;
    // int i;
    // float Alpha[NUMCHAIR + 1];

    // start_all = clock(); // 开始计时
    // // start timing
    // start_all = clock();
    // // reset system
    // sys_reset(Car_Distribution, Req_Distribution, Car_Location);

    // for (i = 0; i < NUMCHAIR + 1; i++)
    // {
    //     Alpha[i] = 0;
    // }

    // // begin execution
    // for (t_interval = 0; t_interval < 1440; t_interval++)
    // {
    //     execution(Alpha, t_interval, Car_Distribution, Req_Distribution, Car_Location, Reward);
    // }

    // // output the results
    // // output(ReqList_ReqTime, ReqList_UpNode, ReqList_OffNode, CarState_NodeID, CarState_Time, CarState_CostToNode, CarState_FirstTask, CarState_ChairNum, MinCost, TaskList_ReqId, TaskList_Node, ReqList_UpTaskId, ReqList_OffTaskId, TaskList_NextTaskID, Car_RouteList, TaskList_WaitTimeOrDetourTime, TaskList_FinishTime, TaskList_FinishTime_Temp, TaskList_Chair, TaskChain, TaskChain_Candi, SysIncreCost, SysIncreRevd, SysIncreWait, SysIncreDetr);
    // output();
    // // end timing
    // finish_all = clock();
    // Total_time = (float)(finish_all - start_all) / CLOCKS_PER_SEC; // 单位换算成秒
    // printf("Total Time: %f seconds\n", Total_time);                // 输出时间

    return 0;
}

// reset the system for the beginning of every epoch
void sys_reset(float Car_Distribution[NUMNODE], float Req_Distribution[NUMNODE], float Car_Location[NUMNODE])
{
    // 用于全局状态的初始化输出：车辆分布、请求分布
    int car;
    int req;
    // 请求数据输入
    reqdata_input();
    // 路网数据输入
    mincost_input();
    // 初始化各个数组
    init_arr(Car_Distribution, Req_Distribution, Car_Location); // 数组初始化
    // 请求与任务转换
    trans_Reqid_Taskid();
    // 输入检查
    //  input_check(ReqList_ReqTime, ReqList_UpNode, ReqList_OffNode, MinCost, CarState_NodeID);
    // input_check();

    // 暂时让棋盘格为一维向量，每个值为车辆在该站点分布频数
    for (car = 0; car < NUMCAR; car++)
    {
        Car_Distribution[CarState_NodeID[car]] += 1;
    }

    // 最开始一分钟内请求的分布
    for (req = 0; ReqList_ReqTime[req] == 0; req++)
    {
        Req_Distribution[ReqList_UpNode[req]] += 1;
    }
}

/*
execute the system for 1 min(step):
input: current time(min),    current state
output: vehicle distribution after the step, request distribution of next step, reward of the step
*/
void execution(float Actions[NUMCAR], int time_id, float Car_Distribution[NUMNODE], float Req_Distribution[NUMNODE], float Car_Location[NUMNODE], float reward[1])
{
    int requestid = -1; // initialize the requestid to -1
    int carid;
    int uptaskid, offtaskid;
    int i, j;
    int car, req;
    float action_car;

    // 找出一分钟内对应的请求id    可以在python中完成
    int req_num = 0;            // 存储时间间隔内请求数量
    int exec_reqids[100] = {0}; // 存储指定时间间隔内的请求id
    // 记录当前时间间隔内的请求id和请求数量
    for (j = 0; j < NUMREQ_RUN; j++)
    {
        if (ReqList_ReqTime[j] == time_id)
        {
            exec_reqids[req_num] = j;
            req_num += 1;
        }
    }

    // 根据一分钟内的请求进行更新、匹配
    for (j = 0; j < req_num; j++)
    {
        int num = 0;                    // 附近车辆数
        int near_cars[NUMCAR];          // 附近车辆的ID
        float near_cars_dis[NUMCAR];    // 附近车辆到目标站点的距离
        Min_Dispatch_Cost = 1000000000; // 最小损失时间变量，初始设为最大
        BestCar = -1;                   // 初始最佳车辆ID为-1
        requestid = exec_reqids[j];
        uptaskid = ReqList_UpTaskId[requestid]; // 获取新请求的上下车任务id
        offtaskid = ReqList_OffTaskId[requestid];

        // 遍历所有车辆，更新状态至新请求发出时刻
        for (i = 0; i < NUMCAR; i++)
        {
            carid = i;
            update(requestid, carid);
        }
        // 搜索一定范围内的车辆，获得附近车辆数
        num = find_near_cars(near_cars_dis, near_cars, requestid);
        // 对车辆id按照距离升序排序
        insert_sort(near_cars_dis, near_cars, num);
        // 避免实施匹配的车辆数大于搜索获得的附近车辆数
        if (num > MAX_MATCH_NUM)
            MatchCar_Num = MAX_MATCH_NUM;
        else
            MatchCar_Num = num;

        // 请求匹配：对搜索获得的最近车辆按距离从近到远进行遍历
        for (i = 0; i < MatchCar_Num; i++)
        {
            carid = near_cars[i];
            action_car = Actions[carid];
            match(action_car, requestid, carid); // 尝试匹配车辆和新请求，计算费用成本
            // 如果等待时间和绕行时间均为0，那么说明该匹配方案已经最优，无需遍历其他车辆
            if (Min_Dispatch_Cost == 0)
            {
                break;
            }
        }

        // 当所有车辆都遍历完毕，当该请求成功匹配到车辆的情况下，在将candi赋值给最终数组进行更新
        if (BestCar > -1)
        {                                         // 仅当匹配到车辆时更新
            Car_RouteList[BestCar][uptaskid] = 1; // 将匹配成功的车辆与任务记录到任务清单中
            Car_RouteList[BestCar][offtaskid] = 1;
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

    // 函数返回：全局共享状态的车辆分布和请求分布 + 实际奖励
    // 状态数组清零
    for (j = 0; j < NUMNODE; j++)
    {
        Car_Distribution[j] = 0;
        Req_Distribution[j] = 0;
        Car_Location[j] = 0;
    }
    // a. 车辆分布
    for (car = 0; car < NUMCAR; car++)
    {
        Car_Distribution[CarState_NodeID[car]] += 1;
    }
    // b. 下一时间间隔的请求分布
    for (req = 0; req < req_num; req++)
    {
        Req_Distribution[ReqList_UpNode[exec_reqids[req]]] += 1;
    }
    // c. 当前时间间隔内的实际奖励
    reward[0] = 0; // 在输入环境前将1个时间步的累计奖励清零
    for (j = 0; j < req_num; j++)
    {
        requestid = exec_reqids[j];
        reward[0] += SysIncreRevd[requestid] - SysIncreCost[requestid] - SysIncreDetr[requestid] - SysIncreWait[requestid];
        // printf("interval%d's reward: %3f\n", ReqList_ReqTime[requestid], reward[0]);
        // printf("check SysIncreCost[%d]: %.3f\n", requestid, SysIncreCost[requestid]);
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

    // output_check();                                 //输出检查
}

// 输出车辆独立的状态    用不用输入时间？？？
// 位置
void car_state_generate(int car_id, float Car_Location[NUMNODE])
{
    int node;

    for (node = 0; node < NUMNODE; node++)
    {
        if (CarState_NodeID[car_id] == node)
            Car_Location[node] = 1;
        else
            Car_Location[node] = 0;
    }
}
