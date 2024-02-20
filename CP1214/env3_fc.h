#ifndef ENV3_FC_H
#define ENV3_FC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "env2_fc.h"

/*
Created on Wednesday October 26 2022
@author: lyj

此模块包括仿真执行模块的基础函数：
update(), match(), find_near_cars(), insert_sort(), output_check(), output()
*/

/*获得当前请求时刻的某出租车辆状态（时间）和系统状态（费用）*/
void update(int req_id, int car_id)
{
    // clock_t start, finish;      //用于统计程序运行时间
    int curnode; /*设置车辆的已完成状态变量：当前位置（节点）、当前状态时间、当前距离上一节点的时间*/
    float curcosttonode, curtime;
    int forwardtaskid; /*设置目标任务id  */
    int forwardnode;   /*设置要完成状态变量：目标位置（节点）、完成目标任务的时间*/
    float forwardtime;
    int reqid_fromtask;        /*用于将任务id转换为对应的请求id的变量*/
    float time_difference = 0; /*提前完成任务的时间差*/
    int chair_num;             /*车上已有乘客数*/

    float car_incre_cost = 0; /*该车辆增加的cost，燃油费用--支出*/
    float car_incre_revd = 0; /*该车辆增加的reward,乘客付费--收入*/
    float car_incre_wait = 0; /*该车辆增加的等待费用，元*/
    float car_incre_detr = 0; /*该车辆增加的绕行费用，元*/

    int i;
    float precicion = 0.001; // 用于小数点位调整

    curnode = CarState_NodeID[car_id]; /*current状态变量获取*/
    curtime = CarState_Time[car_id];
    curcosttonode = CarState_CostToNode[car_id];
    forwardtaskid = CarState_FirstTask[car_id]; /*首先获取forwardtaskid*/
    chair_num = CarState_ChairNum[car_id];      /*获取车辆已有乘客数*/

    /*该车辆有目标任务时，完成新请求发出时刻之前的所有目标任务*/
    if (forwardtaskid >= 0)
    {
        forwardnode = TaskList_Node[forwardtaskid];                            /*根据taskid获取目标站点*/
        forwardtime = curtime + MinCost[curnode][forwardnode] - curcosttonode; /*计算到达目标站点的时刻：：当前时间+两节点间的最短路程时间-距离已经过节点的路程时间*/

        while (forwardtime <= ReqList_ReqTime[req_id])
        {                          /*需完成请求发出时刻之前所有可以完成的任务*/
            curtime = forwardtime; /*任务完成后，将状态时间向前推进*/

            // 记录forwardtask对应的requestID
            reqid_fromtask = TaskList_ReqId[forwardtaskid];
            // 如果是上车任务，存储车辆的收入（乘客缴费）、等待时间、任务完成时间、任务完成时乘客数
            if (forwardtaskid % 2 == 0)
            {
                // 存储每个请求的车费：假设乘客上车就付费；费用和最短路径成正比
                //  ReqList_Reward[reqid_fromtask] = FEE * SPEED * MinCost[ReqList_UpNode[reqid_fromtask]][ReqList_OffNode[reqid_fromtask]];

                if (chair_num > 1)
                    car_incre_revd = car_incre_revd + (FEE * SPEED * MinCost[ReqList_UpNode[reqid_fromtask]][ReqList_OffNode[reqid_fromtask]]) * SHAREDISCOUNT;
                else
                    car_incre_revd = car_incre_revd + FEE * SPEED * MinCost[ReqList_UpNode[reqid_fromtask]][ReqList_OffNode[reqid_fromtask]]; // 2.25

                // 存储等待时间
                TaskList_WaitTimeOrDetourTime[forwardtaskid] = curtime - ReqList_ReqTime[reqid_fromtask];
                // 四舍五入控制小数位数为3
                TaskList_WaitTimeOrDetourTime[forwardtaskid] = (int)(TaskList_WaitTimeOrDetourTime[forwardtaskid] / precicion + 0.5) * precicion;
                // 计算从上一请求发出至新请求发出时刻，坐该车辆乘客的等待时间之和
                car_incre_wait = car_incre_wait + TaskList_WaitTimeOrDetourTime[forwardtaskid] * WAITCOEF; // 将时间转化为金钱

                // 燃油成本的计算，按照人数计费
                if (chair_num == 0)
                    car_incre_cost = car_incre_cost + FUELCOEF0 * (float)(MinCost[curnode][forwardnode] - curcosttonode);
                else if (chair_num == 1)
                    car_incre_cost = car_incre_cost + FUELCOEF1 * (float)(MinCost[curnode][forwardnode] - curcosttonode);
                else if (chair_num == 2)
                    car_incre_cost = car_incre_cost + FUELCOEF2 * (float)(MinCost[curnode][forwardnode] - curcosttonode);
                else if (chair_num == 3)
                    car_incre_cost = car_incre_cost + FUELCOEF3 * (float)(MinCost[curnode][forwardnode] - curcosttonode);
                else
                    printf("车辆数异常！！！");

                // 存储任务完成时间
                TaskList_FinishTime[forwardtaskid] = curtime;
                // 存储任务完成时乘客数：暂设每个请求上车1人
                chair_num = chair_num + 1;
                TaskList_Chair[forwardtaskid] = chair_num; //   1
            }
            // 如果是下车任务，存储绕行时间、任务完成时间、任务完成时乘客数
            else if (forwardtaskid % 2 == 1)
            {
                // 存储绕行时间：实际乘车时间-直达时间
                TaskList_WaitTimeOrDetourTime[forwardtaskid] = ((curtime - TaskList_FinishTime[ReqList_UpTaskId[reqid_fromtask]]) - MinCost[ReqList_UpNode[reqid_fromtask]][ReqList_OffNode[reqid_fromtask]]);
                // 四舍五入控制小数位数为3
                TaskList_WaitTimeOrDetourTime[forwardtaskid] = (int)(TaskList_WaitTimeOrDetourTime[forwardtaskid] / precicion + 0.5) * precicion;
                // 计算从上一请求发出至新请求发出时刻，坐该车辆乘客的绕行时间之和
                car_incre_detr = car_incre_detr + TaskList_WaitTimeOrDetourTime[forwardtaskid] * DETOURCOEF; // 将时间转化为金钱

                // 燃油成本的计算，按照人数计费
                if (chair_num == 0)
                    car_incre_cost = car_incre_cost + FUELCOEF0 * (float)(MinCost[curnode][forwardnode] - curcosttonode);
                else if (chair_num == 1)
                    car_incre_cost = car_incre_cost + FUELCOEF1 * (float)(MinCost[curnode][forwardnode] - curcosttonode);
                else if (chair_num == 2)
                    car_incre_cost = car_incre_cost + FUELCOEF2 * (float)(MinCost[curnode][forwardnode] - curcosttonode);
                else if (chair_num == 3)
                    car_incre_cost = car_incre_cost + FUELCOEF3 * (float)(MinCost[curnode][forwardnode] - curcosttonode);
                else
                    printf("车辆数异常！！！");

                // 存储任务完成时间
                TaskList_FinishTime[forwardtaskid] = curtime;
                // 存储任务完成时乘客数：暂设每个请求上车1人
                chair_num = chair_num - 1;
                TaskList_Chair[forwardtaskid] = chair_num;
            }
            // 更新站点
            curnode = forwardnode;
            // 到达目标站点的同时，意味着costtonode变为0
            curcosttonode = 0;

            forwardtaskid = TaskList_NextTaskID[forwardtaskid]; /*更新forwardtaskid*/
            if (forwardtaskid >= 0)
            { /*如果仍有目标任务，则更新forward状态变量*/
                forwardnode = TaskList_Node[forwardtaskid];
                forwardtime = curtime + MinCost[curnode][forwardnode] - curcosttonode;
            }
            else
            {                                                        /*如果更新后没有目标任务，退出循环*/
                time_difference = ReqList_ReqTime[req_id] - curtime; // 时间差：车辆提前完成任务的时间，保证车辆在原地待命，curtime <= ReqList_ReqTime[req_id]
                break;
            }
        }
        /*此时要将车辆状态变量（位置、距上一节点时间、首个目标任务）→更新到→新请求发出时刻*/
        CarState_NodeID[car_id] = curnode;
        CarState_CostToNode[car_id] = ReqList_ReqTime[req_id] - (curtime - curcosttonode) - time_difference;
        CarState_FirstTask[car_id] = forwardtaskid;
        CarState_ChairNum[car_id] = chair_num;
    }

    // // check
    // if (car_id == 4)
    // {
    // 	printf("car4 's cost: %.3f\n", car_incre_cost);
    // 	printf("car4 's revd: %.3f\n", car_incre_revd);
    // 	printf("car4 's wait: %.3f\n", car_incre_wait);
    // 	printf("car4 's detr: %.3f\n", car_incre_detr);
    // }
    // printf("MinCost[14][17]: %f\n", MinCost[14][17]);

    // 更新车辆状态时间
    CarState_Time[car_id] = ReqList_ReqTime[req_id]; // 即使车辆没有目标任务，状态时间也会更新
    // 更新该请求累加的支出、收入、等待成本、绕行成本，即该时间间隔（上一请求到新请求）内所有车辆的累计数值
    SysIncreCost[req_id] = SysIncreCost[req_id] + car_incre_cost;
    SysIncreRevd[req_id] = SysIncreRevd[req_id] + car_incre_revd;
    SysIncreWait[req_id] = SysIncreWait[req_id] + car_incre_wait;
    SysIncreDetr[req_id] = SysIncreDetr[req_id] + car_incre_detr;
}

/*状态时间已更新到新请求发出时刻，将新请求插入到车辆的任务清单中*/
void match(float action, int req_id, int car_id)
{
    float waittime, detourtime;              /*单个任务的等待时间与绕行时间*/
    float waittime_all, detourtime_all;      // 处理有两个以上任务车辆时，存储任务链上所有任务的等待时间或绕行时间，用于最优方案选择的判断
    float dispatch_cost = Min_Dispatch_Cost; /*先将调度成本设定为最大，以防未赋值情况*/

    int uptask, upnode;   /*新请求对应的上车任务id，上车节点*/
    int offtask, offnode; /*新请求对应的下车任务id，下车节点*/
    int curnode;          /*该车辆的已完成状态变量*/
    float curcosttonode, curtime;
    int forwardtask, forwardnode;   /*该车辆的目标状态变量*/
    int nexttask;                   /*下一个目标任务的id，用于判断目标任务数量*/
    int fwdtask_temp, fwdnode_temp; /*用于各可行方案的尝试，计算损失时间*/
    float fwdcosttonode_temp;
    int feasible_switch = 1; /*是否能分配*/
    int chair_num;           /*车上乘客数*/
    int reqid_fromtask;      // 用于task和request两者id转换
    int dt, wt;              // 存储累加等待时间或绕行时间的次数，用于获取单个乘客的平均损失时间
    int i, j, m;             // 循环变量
    int uploc = 1;           // uptask插入的位置,一开始插入位置为1
    int offloc;              // offtask插入的位置,一开始插入位置为uptask后1个单位
    int tasknum;             // 存储未完成任务链的数量
    int visitloc;            // 对可能方案的任务链进行计算时，任务访问位置

    // 对于每个车辆，初始化TaskChain列表为-1
    for (i = 0; i < 50; i++)
        TaskChain[i] = -1;
    // 对于每个车辆，初始化FinishTime_Temp为-1
    for (i = 0; i < NUMTASK_RUN; i++) // 在匹配方案计算过程中，完成时间的存储同时采用TaskList_FinishTime和Temp，更快方法？
        TaskList_FinishTime_Temp[i] = -1;

    uptask = ReqList_UpTaskId[req_id];
    upnode = ReqList_UpNode[req_id];
    offtask = ReqList_OffTaskId[req_id];
    offnode = ReqList_OffNode[req_id];
    curnode = CarState_NodeID[car_id];
    curcosttonode = CarState_CostToNode[car_id];
    curtime = CarState_Time[car_id];
    forwardtask = CarState_FirstTask[car_id];
    if (forwardtask >= 0)
        forwardnode = TaskList_Node[forwardtask]; // 否则arr[-1],超出存储空间！
    chair_num = CarState_ChairNum[car_id];

    // 尝试新请求两个任务的插入
    // 当该车辆没有目标任务时，直接插入两个任务
    if (forwardtask < 0)
    {
        // 新任务插入
        TaskChain[0] = uptask;
        TaskChain[1] = offtask;

        // 计算乘客损失时间
        waittime = MinCost[curnode][upnode];            // 车辆无任务执行时，默认其停留在curnode节点，直至接到下一个任务     2
        detourtime = 0;                                 // 没有合乘过程，所以没有绕行时间
        dispatch_cost = waittime + detourtime + action; // 损失时间为两者之和

        // 判断等待时间是否超出乘客等待容忍时间上限
        if (waittime > WAITTHRESHOLD)
            feasible_switch = 0;
        // 判断是是否需要将Temp写入Candidate
        if (dispatch_cost < Min_Dispatch_Cost && feasible_switch == 1)
        {
            // 更新最小调度成本
            Min_Dispatch_Cost = dispatch_cost;
            // 更新最优车辆ID，用于更新任务清单
            BestCar = car_id;
            memcpy(TaskChain_Candi, TaskChain, NUM_TaskChain);
        }
    }
    // 当该车辆有目标任务
    else
    {
        nexttask = TaskList_NextTaskID[forwardtask];
        // 该车辆只有一个目标任务，该任务必为下车任务
        if (nexttask < 0)
        {
            // 任务插入，forward→up→off
            TaskChain[0] = forwardtask;
            TaskChain[1] = uptask;
            TaskChain[2] = offtask;
            // 计算新上车任务的等待时间
            // 请求发出时刻地点到目标节点最短时间+目标节点到新上车地点最短时间
            waittime = MinCost[curnode][forwardnode] - curcosttonode + MinCost[forwardnode][upnode];
            detourtime = 0; // 没有合乘过程，新下车任务的绕行时间为0
            dispatch_cost = waittime + detourtime + action;

            // 判断等待时间是否超出乘客等待容忍时间上限
            if (waittime > WAITTHRESHOLD)
                feasible_switch = 0;
            // 判断是是否需要将Temp写入Candidate
            if (dispatch_cost < Min_Dispatch_Cost && feasible_switch == 1)
            {
                // 更新最小损失时间
                Min_Dispatch_Cost = dispatch_cost;
                // 更新最优车辆ID，用于更新任务清单
                BestCar = car_id;
                memcpy(TaskChain_Candi, TaskChain, NUM_TaskChain);
            }
        }
        // 该车辆有两个及以上目标任务,该情况下需加入座位数约束
        else
        {
            // 对于每一辆车，计算未完成任务的数量（>=2）                        //或者保存未完成任务链至TaskChain，建立Temp
            TaskChain[0] = CarState_FirstTask[car_id];
            for (i = 0; TaskChain[i] != -1; i++)
            {
                TaskChain[i + 1] = TaskList_NextTaskID[TaskChain[i]];
            }
            tasknum = i; // 结束后i的值为TaskChain中的任务数（不为-1的元素数量）
            for (uploc = 1; uploc <= tasknum; uploc++)
            { // 外部循环，后移uploc
                for (offloc = (uploc + 1); offloc <= (tasknum + 1); offloc++)
                { // 内层循环，后移offloc
                    // 对于每一种可能的插入情况，插入前重新拷贝原任务链
                    TaskChain[0] = CarState_FirstTask[car_id];
                    for (i = 0; TaskChain[i] != -1; i++)
                    {
                        TaskChain[i + 1] = TaskList_NextTaskID[TaskChain[i]];
                    }
                    // 新任务插入，遍历所有插入情况
                    for (j = i - 1; j >= uploc; j--)
                    { // 将插入位置之后的元素向后移
                        TaskChain[j + 1] = TaskChain[j];
                    }
                    TaskChain[uploc] = uptask; // 插入上车任务
                    i = i + 1;                 // TaskChain中的任务数增加1个
                    for (j = i - 1; j >= offloc; j--)
                    {
                        TaskChain[j + 1] = TaskChain[j];
                    }
                    TaskChain[offloc] = offtask;

                    // 初始化各状态变量
                    curnode = CarState_NodeID[car_id]; // 尝试每个方案时，currrent为车辆在新请求发出时刻的状态
                    curcosttonode = CarState_CostToNode[car_id];
                    curtime = CarState_Time[car_id];
                    visitloc = 0;                // 初始访问位置为0
                    fwdtask_temp = TaskChain[0]; // 初始forwardtask为任务链上的第一个任务
                    fwdnode_temp = TaskList_Node[fwdtask_temp];
                    fwdcosttonode_temp = 0;
                    chair_num = CarState_ChairNum[car_id]; // 车上乘客数
                    waittime = 0;
                    detourtime = 0;
                    waittime_all = 0;
                    detourtime_all = 0;
                    wt = 0;
                    dt = 0;
                    feasible_switch = 1;
                    // 计算损失时间
                    while (fwdtask_temp >= 0)
                    {
                        fwdnode_temp = TaskList_Node[fwdtask_temp]; // 确定forwardtask不为-1后再获取node
                        curtime = curtime + MinCost[curnode][fwdnode_temp] - curcosttonode;

                        // 如果是上车
                        if ((fwdtask_temp % 2) == 0)
                        {
                            reqid_fromtask = TaskList_ReqId[fwdtask_temp];    // 记录fwdtask_temp对应的requestID
                            TaskList_FinishTime_Temp[fwdtask_temp] = curtime; // 记录该上车任务完成时间，在下车绕行时间的计算中需要
                            // 计算该上车任务的等待时间
                            waittime = curtime - ReqList_ReqTime[reqid_fromtask];
                            // 存储任务链的累加等待时间
                            waittime_all = waittime_all + waittime;
                            wt = wt + 1; // 累加次数+1

                            // 判断单个上车任务的等待时间是否超限
                            if (waittime > WAITTHRESHOLD)
                            {
                                feasible_switch = 0;
                                break; // 如果某任务出现超限，直接跳出该种插入情况下的visit任务迭代，即该种插入情况不可行
                            }
                            // 车上乘客数+1，并进行座位数判断
                            chair_num += 1;
                            if (chair_num > NUMCHAIR)
                            {
                                feasible_switch = 0;
                                break;
                            }
                        }
                        // 如果是下车
                        else
                        {
                            reqid_fromtask = TaskList_ReqId[fwdtask_temp]; // 记录fwdtask_temp对应的requestID
                            // 计算该下车任务的绕行时间
                            if (TaskList_FinishTime_Temp[ReqList_UpTaskId[reqid_fromtask]] >= 0)
                                detourtime = (curtime - TaskList_FinishTime_Temp[ReqList_UpTaskId[reqid_fromtask]]) - MinCost[ReqList_UpNode[reqid_fromtask]][ReqList_OffNode[reqid_fromtask]];
                            else
                                detourtime = (curtime - TaskList_FinishTime[ReqList_UpTaskId[reqid_fromtask]]) - MinCost[ReqList_UpNode[reqid_fromtask]][ReqList_OffNode[reqid_fromtask]];
                            // 存储任务链的累加绕行时间
                            detourtime_all = detourtime_all + detourtime;
                            // 累加次数+1
                            dt = dt + 1;
                            // 判断是否超限
                            if (detourtime > DETOURTHRESHOLD)
                            {
                                feasible_switch = 0;
                                break;
                            }
                            // 车上乘客数-1
                            chair_num -= 1;
                        }
                        // 将站点和costtonode更新至完成forwardtask时刻
                        curnode = fwdnode_temp;
                        curcosttonode = fwdcosttonode_temp;

                        visitloc++;                         // 任务链访问位置后移1
                        fwdtask_temp = TaskChain[visitloc]; // 按照假定插入情况（temp）更新
                    }

                    // 该种插入情况下所有待完成任务的等待时间和绕行时间之和
                    dispatch_cost = (waittime_all / wt) + (detourtime_all / dt) + action; // 由于按照每个请求的平均等待/平均绕行，可能出现有的请求等待时间很长
                    // 判断是否需要将temp写入candi
                    if ((dispatch_cost <= Min_Dispatch_Cost) && (feasible_switch == 1))
                    {
                        // 更新最小损失时间
                        Min_Dispatch_Cost = dispatch_cost;
                        // 更新最优车辆ID
                        BestCar = car_id;
                        // 更新任务链
                        memcpy(TaskChain_Candi, TaskChain, NUM_TaskChain);
                    }
                }
            }
        }
    }
}

// 搜索新请求上车站点附近一定范围内车辆
int find_near_cars(float near_cars_dis[], int near_cars[], int reqid)
{
    int i, carid; // 循环变量
    int n = 0;    // 搜索得到的车辆数
    // 对于每个请求，搜索前清零
    for (i = 0; i < NUMCAR; i++)
    {
        near_cars_dis[i] = -1;
    }
    for (i = 0; i < NUMCAR; i++)
    {
        near_cars[i] = -1;
    }
    // 开始搜索
    for (i = 0; i < NUMCAR; i++)
    {
        carid = i;
        if (MinCost[CarState_NodeID[carid]][ReqList_UpNode[reqid]] <= (WAITTHRESHOLD + 10))
        {
            near_cars[n] = carid;                                                      // 获取符合距离条件内的车辆ID
            near_cars_dis[n] = MinCost[CarState_NodeID[carid]][ReqList_UpNode[reqid]]; // 存储到目标站点距离
            n++;                                                                       // 附近车数+1
        }
    }
    return n;
}

// 插入排序法：对搜索获得的附近车辆按照距离升序排序
void insert_sort(float distance[], int carids[], int a)
{
    int i, j, mm; // 循环变量

    for (i = 1; i < a; i++)
    {
        // 用每个车辆id对应的距离作为排序依据
        if (distance[i] < distance[i - 1])
        { // 若第 i 个元素大于 i-1 元素则直接插入；反之，需要找到适当的插入位置后在插入。
            j = i - 1;
            int idx = carids[i];      // 保存哨兵id
            float disx = distance[i]; // 保存哨兵distance
            while (j > -1 && disx < distance[j])
            { // 采用顺序查找方式找到插入的位置，在查找的同时，将数组中的元素进行后移操作，给插入元素腾出空间
                distance[j + 1] = distance[j];
                carids[j + 1] = carids[j];
                j--;
            }
            distance[j + 1] = disx;
            carids[j + 1] = idx; // 插入到正确位置
        }
        // printf("第%d次排序结果：", i);
        // for (mm = 0; mm < a; mm++)
        //	printf("%d ", carids[mm]);//打印每次排序后的结果
        // printf("\n");
    }
}

// 仿真器输出数据的打印检查
void output_check()
{
    int i, j; // 循环变量
    printf("****************运行结束，开始检查******************\n");
    // 检验任务清单
    //  printf("Car_RouteList:\n");
    //  for (j = 0; j < NUMCAR; j++) {
    //	for (i = 0; i < NUMTASK_RUN; i++) {
    //		printf("%d ", Car_RouteList[j][i]);
    //	}
    //	printf("\n");
    // }
    // 检验任务完成时车辆人数
    printf("TaskList_Chair:\n");
    for (i = 0; i < NUMTASK_RUN; i++)
    {
        printf("%d \n", TaskList_Chair[i]);
    }
    printf("\n");
    // 检验任务完成时间
    printf("TaskList_FinishTime:\n");
    for (i = 0; i < NUMTASK_RUN; i++)
    {
        printf("%f \n", TaskList_FinishTime[i]);
    }
    printf("\n");
    // 检验各任务的乘客损失时间
    printf("TaskList_WaitTimeOrDetourTime:\n");
    for (i = 0; i < NUMTASK_RUN; i++)
    {
        printf("%f \n", TaskList_WaitTimeOrDetourTime[i]);
    }
    printf("\n");
    // 检验各请求的费用、成本
    //  printf("截止该请求时刻的时段内系统收入、成本、乘客时间:\n");
    //  for (i = 0; i < NUMREQ_RUN; i++) {
    //	printf("请求%d：cost = %f, ", i, SysIncreCost[i]);
    //	printf("reward = %f, ", SysIncreRevd[i]);
    //	printf("wait = %d, ", SysIncreWait[i]);
    //	printf("detour = %d, \n", SysIncreDetr[i]);
    // }
    //  printf("\n");
}

// 将运行结果导出到csv文件中进行保存
void output()
{
    int i = 0, j = 0;
    // 创建文件指针
    FILE *fp_RouteList = fopen("/home/chwei/JeanLu/SharedTaxi/C_python/dataoutput/Alpha1207/Car_RouteList_500.csv", "w+");
    FILE *fp_Chair = fopen("/home/chwei/JeanLu/SharedTaxi/C_python/dataoutput/Alpha1207/TaskList_Chair_500.csv", "w+");
    FILE *fp_FinishTime = fopen("/home/chwei/JeanLu/SharedTaxi/C_python/dataoutput/Alpha1207/TaskList_FinishTime_500.csv", "w+");
    FILE *fp_Cost_Reward = fopen("/home/chwei/JeanLu/SharedTaxi/C_python/dataoutput/Alpha1207/ReqList_Cost_Reward_500.csv", "w+");
    FILE *fp_WaitTimeOrDetourTime = fopen("/home/chwei/JeanLu/SharedTaxi/C_python/dataoutput/Alpha1207/TaskList_WaitTimeOrDetourTime_500.csv", "w+");
    FILE *fp_NextTask = fopen("/home/chwei/JeanLu/SharedTaxi/C_python/dataoutput/Alpha1207/TaskList_NextTaskID_500.csv", "w+");
    // 出错处理,避免传入空指针
    if ((fp_RouteList == NULL) || (fp_Chair == NULL) || (fp_FinishTime == NULL) || (fp_Cost_Reward == NULL) || (fp_WaitTimeOrDetourTime == NULL) || (fp_NextTask == NULL))
    {
        printf("打开错误...");
        return;
    }
    // 【写入任务清单】
    //  for (i = 0; i < NUMCAR; i++)
    //{
    //	for (j = 0; j < NUMTASK_RUN; j++)
    //	{
    //		fprintf(fp_RouteList, "%d,", Car_RouteList[i][j]);//把每个数组元素以十进制整数的方式存入csv中
    //	}
    //	fprintf(fp_RouteList, "\n");
    // }
    //  fclose(fp_RouteList);//关闭文件
    // 【写入任务完成时的乘客数】
    fprintf(fp_Chair, "TaskID,FinishTime_Chairs\n");
    for (i = 0; i < NUMTASK_RUN; i++)
    {
        fprintf(fp_Chair, "%d,%d\n", i, TaskList_Chair[i]);
    }
    fclose(fp_Chair); // 关闭文件
    // 【写入任务完成时间】
    fprintf(fp_FinishTime, "TaskID,FinishTime\n");
    for (i = 0; i < NUMTASK_RUN; i++)
    {
        fprintf(fp_FinishTime, "%d,%f\n", i, TaskList_FinishTime[i]);
    }
    fclose(fp_FinishTime); // 关闭文件
    // 【写入系统费用、成本】
    fprintf(fp_Cost_Reward, "RequestID,Cost,Reward,Wait_Time,Detour_Time\n");
    for (i = 0; i < NUMREQ_RUN; i++)
    {
        fprintf(fp_Cost_Reward, "%d,%f,%f,%f,%f\n", i, SysIncreCost[i], SysIncreRevd[i], SysIncreWait[i], SysIncreDetr[i]);
    }
    fclose(fp_Cost_Reward); // 关闭文件
    // 【写入每个任务的损失时间】
    fprintf(fp_WaitTimeOrDetourTime, "TaskID,WaitTimeOrDetourTime\n");
    for (i = 0; i < NUMTASK_RUN; i++)
    {
        fprintf(fp_WaitTimeOrDetourTime, "%d,%f\n", i, TaskList_WaitTimeOrDetourTime[i]);
    }
    fclose(fp_WaitTimeOrDetourTime); // 关闭文件fp_NextTask
    // 【写入任务链】
    fprintf(fp_NextTask, "TaskID,NextTaskID\n");
    for (i = 0; i < NUMTASK_RUN; i++)
    {
        fprintf(fp_NextTask, "%d,%d\n", i, TaskList_NextTaskID[i]);
    }
    fclose(fp_NextTask); // 关闭文件
}

#endif