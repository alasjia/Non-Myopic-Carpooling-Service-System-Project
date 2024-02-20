#ifndef ENV1_CNN_H
#define ENV1_CNN_H

#include <stdio.h>
#include <stdlib.h>

/*
Created on Wednesday October 26 2022
@author: lyj

此模块用于定义各类常量和外部变量，并进行函数声明
*/

/*定义各类常量*/
#define NUMREQ 48490      // 49062 = 25000 + 24062  /*请求次数,18-19为3550*/
#define NUMREQ_RUN 48490  // 实际运行的请求数
#define NUMCAR 500        /*系统中的车辆数*/
#define NUMTASK 96980     // 98124 = 50000 + 48124 /*任务数量,一个请求包括上车、下车两个任务   18-19为7100*/
#define NUMTASK_RUN 96980 // 实际运行的任务数
#define NUMNODE 880       /*gird站点数量，列数*/
#define WAITTHRESHOLD 10  /*容忍等待时间,min*/
#define DETOURTHRESHOLD 5 /*容忍绕路时间,min*/
#define FEE 3             /*单位为【元/km】，乘客上车之后的收费*/

#define SHAREDISCOUNT 0.8 /*合乘出行的费用折扣系数*/

#define SPEED 0.25 /* 速度，单位为【km/min】，即15km/h*/

#define FUELCOEF0 0.1   /*乘客数为0时，单位时间燃油成本，单位为【元/min】，换算自0.4元/km */
#define FUELCOEF1 0.125 /*乘客数为1时，单位时间燃油成本，单位为【元/min】，换算自0.5元/km */
#define FUELCOEF2 0.15  /*乘客数为2时，单位时间燃油成本，单位为【元/min】，换算自0.6元/km */
#define FUELCOEF3 0.175 /*乘客数为3时，单位时间燃油成本，单位为【元/min】，换算自0.7元/km */
// 增加一个乘客，油耗费用增加约0.025元/km

#define WAITCOEF 0.125     /*等待成本换算系数，单位为【元/min】，换算自0.5元/km */
#define DETOURCOEF 0.75    /* 绕行成本换算系数，单位为【元/min】,换算自3元/km */
#define NUMCHAIR 3         /* 车辆最大座位数 */
#define MAX_MATCH_NUM 100  /*遍历的最大附近车辆数*/
#define NUM_TaskChain 4000 /*200 = 4*50，4是每个int类型占据的字节数  数据类型占据的字节数和编译器有关*/
/*注意：memcpy()中以sizeof(array)的方式传参会造成  sizeof(array) = size of (int* array[0]) */

// /*定义外部变量*/
// float Alpha[NUMCAR]; /*RL学习的决策内容，车辆属性值*/

int MatchCar_Num;        /*附近车辆按距离升序排序后，与新请求进行匹配操作的车辆个数*/
float Min_Dispatch_Cost; /*最小损失时间变量*/
int BestCar;             /*每个请求匹配到的最佳车辆ID*/

/* Input:
乘客信息：请求到达时间，上下车任务ID（两个），上下车地点（均已请求ID作为索引）
车辆信息：初始地点，初始状态时间，初始 车辆距离上一个节点的行驶时间，座位数，当前任务
路网信息：各站点间最小出行时间
*/
// Request
int ReqList_ReqTime[NUMREQ]; /*请求到达时间*/
int ReqList_UpNode[NUMREQ];  /*请求对应的上车地点*/
int ReqList_OffNode[NUMREQ]; /*请求对应的下车地点*/
// Vehicle
int CarState_NodeID[NUMCAR];       /*记录车辆状态中的地点，初始地点*/
int CarState_Time[NUMCAR];         /*记录车辆状态中的更新时间，初始状态时间均为0,单位为分钟*/
float CarState_CostToNode[NUMCAR]; /*记录车辆距离上一个节点的行驶时间，初始均为0，因为均在节点上*/
int CarState_ChairNum[NUMCAR];     /*车辆中乘坐的乘客数量，初始均为0*/
int CarState_FirstTask[NUMCAR];    /*车辆当前未完成的第一个任务，初始均为-1*/
// Network
float MinCost[NUMNODE][NUMNODE]; /*各站点间最小出行时间,单位为min*/

/*中间需求变量列表：*/
int TaskList_ReqId[NUMTASK];   /*以任务id为索引，任务id对应的请求id*/
int TaskList_Node[NUMTASK];    /*任务id对应的目标节点*/
int ReqList_UpTaskId[NUMREQ];  /*请求id对应的上车任务id，偶数*/
int ReqList_OffTaskId[NUMREQ]; /*请求id对应的下车任务id，奇数*/
int Nodes_Manhattan[225];      /*曼哈顿研究区域内站点ID*/
int Fixed_Arr[NUMCAR];
int CarState_Node_15[NUMCAR]; /*记录无请求到达推演15mins、30mins后的车辆位置id*/
int CarState_Node_30[NUMCAR];

/*Output: */
int TaskList_NextTaskID[NUMTASK]; /*任务链：每个任务按时间顺序的下一个任务，任务的实际执行顺序,初始均为-1*/
// int Car_RouteList[NUMCAR][NUMTASK];           /*任务分配结果：每个车辆的任务清单，没有任务为-1，任务由该车完成为1，初始均为-1*/
float TaskList_WaitTimeOrDetourTime[NUMTASK]; /*完成每个任务的损失时间，上车任务为等待时间，下车任务为绕行时间，初始均为-1*/
int TaskList_Chair[NUMTASK];                  /*存储完成任务时车上的乘客数，初始均为-1*/
float TaskList_FinishTime[NUMTASK];           /*存储任务的完成时间，初始均为-1*/

float SysIncreCost[NUMREQ]; /*系统累计燃油支出*/
float SysIncreRevd[NUMREQ]; /*系统累计车费收入*/
float SysIncreWait[NUMREQ]; /*系统累计等待时间*/
float SysIncreDetr[NUMREQ]; /*系统累计绕行时间*/

/*Temporary: */
float TaskList_FinishTime_Temp[NUMTASK];
// 部分任务链列表，存储某车辆在匹配新请求时可能的实时任务链
int TaskChain[1000];       // 存储可能的任务链，每个车辆初始化一次           50!!!
int TaskChain_Candi[1000]; // 存储最佳车辆匹配后的任务链

// // 单个步长（时间间隔）返回的实际奖励
// float Reward[1];

/*Function Prototype*/
void reqdata_input();
void mincost_input();
void init_arr(float Output_C_Distribution[NUMNODE], float Output_C_Distribution_15[NUMNODE], float Output_C_Distribution_30[NUMNODE], float Output_R_Distribution[NUMNODE], float Output_R_Distribution_15[NUMNODE], float Output_R_Distribution_30[NUMNODE], float Output_C_Location[NUMNODE]);
void trans_Reqid_Taskid();
int find_near_cars(float distance[], int carids[], int reqid);
void insert_sort(float distance[], int carids[], int a);
void update(int req_id, int car_id);
void match(float action_car, int req_id, int car_id);
void nodes_in_manhattan_fixed();

void update_15(int time_id, int car_id);
void update_30(int time_id, int car_id);
// Checking Function
void input_check();
void output_check();
void output();
// Superior Function
void sys_reset(float Output_C_Distribution[NUMNODE], float Output_C_Distribution_15[NUMNODE], float Output_C_Distribution_30[NUMNODE], float Output_R_Distribution[NUMNODE], float Output_R_Distribution_15[NUMNODE], float Output_R_Distribution_30[NUMNODE], float Output_C_Location[NUMNODE]);
void execution(float Actions[NUMCHAIR + 1], int time_id, float Output_C_Distribution[NUMNODE], float Output_C_Distribution_15[NUMNODE], float Output_C_Distribution_30[NUMNODE], float Output_R_Distribution[NUMNODE], float Output_R_Distribution_15[NUMNODE], float Output_R_Distribution_30[NUMNODE], float Output_C_Location[NUMNODE], float Output_Reward[1]);

#endif