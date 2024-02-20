#ifndef ENV2_CNN_H
#define ENV2_CNN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "env1_cnn.h"

/*
Created on Wednesday October 26 2022
@author: lyj

此模块包括初始化模块的基础函数：
reqdata_input,  mincost_input, init_arr, trans_Reqid_Taskid,
*/

/*将包含请求数据的csv文件输入数组*/
void reqdata_input()
{
    FILE *fp = NULL;
    char *line, *record;
    char buffer[1024];
    char delims[] = ",";
    int toint;
    int i = 0; // 行号
    int j = 0; // 列号
    if ((fp = fopen("/home/chwei/JeanLu/TaxiData1215/request/Requests.csv", "r")) != NULL)
    {
        line = fgets(buffer, sizeof(buffer), fp);                  // 不打印第一行表头
        while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) // 当没有读取到文件末尾时循环继续
        {
            // printf("%s \n", line);
            record = strtok(line, delims); // 初步读取每行第一列的值
            while (record != NULL)         // 读取每一行的数据
            {
                toint = atoi(record); // 将每个值从字符型转换为整型
                // printf("%s\n", record);
                switch (j)
                { // 根据列号判断相应的request信息
                case 0:
                    ReqList_ReqTime[i] = toint; // 第一列为请求时间
                    // printf("%d\n", ReqList_ReqTime[i]);
                    break;
                case 1:
                    ReqList_UpNode[i] = toint; // 第二列为请求上车地点
                    break;
                case 2:
                    ReqList_OffNode[i] = toint; // 第三列为请求下车地点
                    break;
                default:
                    break; // 其余列号则跳出switch循环
                }
                record = strtok(NULL, delims); /*record重新储存列表行中下一个值*/
                j++;                           // 列号加一
            }
            i++; // 行号加一
            j = 0;
        }
        fclose(fp);
        fp = NULL;
    }
}

/*将包含路网数据的csv文件输入数组*/
void mincost_input()
{
    FILE *fp = NULL;
    char *line, *record;
    char buffer[20000];
    char delims[] = ",";
    float tofloat;
    int i = 0; // 行号
    int j = 0; // 列号
    if ((fp = fopen("/home/chwei/JeanLu/TaxiData1215/network/MinTime1215_lowspeed.csv", "r+")) != NULL)
    {
        while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) // 当没有读取到文件末尾时循环继续
        {
            // printf("第%d行 %s \n", i, line);
            record = strtok(line, delims); // 初步读取每行第一列的值
            while (record != NULL)         // 读取每一行的数据
            {
                tofloat = atof(record); // 将每个值从字符型转换为
                MinCost[i][j] = tofloat;
                // printf("%f ", MinCost[i][j]);
                record = strtok(NULL, delims); /*record重新储存列表行中下一个值*/
                j++;                           // 列号加一
            }
            // printf("\n");
            i++; // 行号加一
            j = 0;
        }
        fclose(fp);
        fp = NULL;
    }
}

/*将曼哈顿内站点ID的csv文件输入数组，并创建车辆初始分布位置数组*/
void nodes_in_manhattan_fixed()
{
    FILE *fp = NULL;
    char *line, *record;
    char buffer[1024];
    char delims[] = ",";
    int toint;
    int i = 0; // 行号
    int j = 0; // 列号
    if ((fp = fopen("/home/chwei/JeanLu/TaxiData1215/NodesID_in_Manhattan.csv", "r")) != NULL)
    {
        line = fgets(buffer, sizeof(buffer), fp);                  // 不打印第一行表头
        while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) // 当没有读取到文件末尾时循环继续
        {
            // printf("%s \n", line);
            record = strtok(line, delims); // 初步读取每行第一列的值
            while (record != NULL)         // 读取每一行的数据
            {
                toint = atoi(record); // 将每个值从字符型转换为整型
                // printf("%s\n", record);
                if (j == 0)
                    Nodes_Manhattan[i] = toint; // 第一列为ID号
                record = strtok(NULL, delims);  /*record重新储存列表行中下一个值*/
                j++;                            // 列号加一
            }
            i++; // 行号加一
            j = 0;
        }
        fclose(fp);
        fp = NULL;
    }

    int m;
    for (m = 0; m < NUMCAR; m++)
    {
        Fixed_Arr[m] = Nodes_Manhattan[rand() % 225];
    }
}

/*初始化中间变量数组和输出变量数组*/
void init_arr(float Car_Distribution[NUMNODE], float Car_Distribution_15[NUMNODE], float Car_Distribution_30[NUMNODE], float Req_Distribution[NUMNODE], float Req_Distribution_15[NUMNODE], float Req_Distribution_30[NUMNODE], float Car_Location[NUMNODE])
{
    int i;
    int j;
    for (i = 0; i < NUMCAR; i++)
    {
        CarState_FirstTask[i] = -1;
        CarState_ChairNum[i] = 0; // 可写可不写，默认整型数组初始为0
        CarState_Time[i] = 0;
        CarState_CostToNode[i] = 0;
        // CarState_NodeID[i] = Nodes_Manhattan[rand() % 225];
        CarState_NodeID[i] = Fixed_Arr[i];
        CarState_Node_15[i] = CarState_NodeID[i];
        CarState_Node_30[i] = CarState_NodeID[i];
    }
    for (i = 0; i < NUMTASK_RUN; i++)
    {
        TaskList_FinishTime[i] = -1;
        TaskList_FinishTime_Temp[i] = -1;
        TaskList_NextTaskID[i] = -1;
        TaskList_WaitTimeOrDetourTime[i] = -1;
        TaskList_Chair[i] = -1;
    }
    // for (i = 0; i < NUMCAR; i++)
    // {
    //     for (j = 0; j < NUMTASK_RUN; j++)
    //     {
    //         Car_RouteList[i][j] = -1;
    //     }
    // }
    for (i = 0; i < NUMREQ_RUN; i++)
    {
        SysIncreCost[i] = 0;
        SysIncreRevd[i] = 0;
        SysIncreWait[i] = 0;
        SysIncreDetr[i] = 0;
    }

    for (i = 0; i < NUMNODE; i++)
    {
        Car_Distribution[i] = 0;
        Car_Distribution_15[i] = 0;
        Car_Distribution_30[i] = 0;
        Req_Distribution[i] = 0;
        Req_Distribution_15[i] = 0;
        Req_Distribution_30[i] = 0;
        Car_Location[i] = 0;
    }
}

/*任务与请求的互相转换、列表存储*/
void trans_Reqid_Taskid()
{
    /*获得四个列表：
    TaskList_ReqId[]  TaskList_Node[]
    ReqList_UpTaskId[]  ReqList_OffTaskId[]
    */
    int req_id; /*设置循环变量*/
    int uptaskid, offtaskid;
    for (req_id = 0; req_id < NUMREQ_RUN; req_id++)
    {
        uptaskid = req_id * 2;      /* 获得将该请求所对应的上车任务ID*/
        offtaskid = req_id * 2 + 1; /* 获得将该请求所对应的下车任务ID*/

        TaskList_ReqId[uptaskid] = req_id;                /*存储上车任务所对应的请求ID*/
        TaskList_Node[uptaskid] = ReqList_UpNode[req_id]; /*存储上车任务所对应的上车站点ID*/

        TaskList_ReqId[offtaskid] = req_id;                 /*存储下车任务所对应的请求ID*/
        TaskList_Node[offtaskid] = ReqList_OffNode[req_id]; /*存储下车任务所对应的下车站点ID*/

        ReqList_UpTaskId[req_id] = uptaskid;   /*存储该请求的上车任务的ID*/
        ReqList_OffTaskId[req_id] = offtaskid; /*存储该请求的下车任务的ID*/
    }
}

// 仿真器输入数据的打印检查
void input_check()
{
    int i, j; // 循环变量
    // // 请求数据的输入检验
    // printf("ReqTime UpNode OffNode:\n");
    // for (i = 0; i < NUMREQ_RUN; i++)
    // {
    //     printf("%d        ", ReqList_ReqTime[i]);
    //     printf("%d        ", ReqList_UpNode[i]);
    //     printf("%d", ReqList_OffNode[i]);
    //     printf("\n");
    // }

    // // 路网节点间最短行驶时间的检验
    // printf("Network minimum time:\n");
    // for (i = 0; i < NUMNODE; i++)
    // {
    //     for (j = 0; j < NUMNODE; j++)
    //     {
    //         printf("%f ", MinCost[i][j]);
    //     }
    //     printf("\n");
    // }

    // // 查看所有输入请求的平均OD行驶时间
    // float meannum = 0;
    // for (i = 0; i < NUMREQ_RUN; i++)
    // {
    //     meannum += MinCost[ReqList_UpNode[i]][ReqList_OffNode[i]];
    // }
    // meannum = meannum / i;
    // printf("OD average time: \n");
    // printf("%f\n", meannum); // 9.24 min

    // // 初始化检验
    // printf("SysIncreDetr:\n");
    // for (i = 0; i < NUMREQ_RUN; i++)
    // { // 可检验每个任务的乘客数、完成时间、成本费用等
    //     printf("%f \n", SysIncreDetr[i]);
    // }

    // // int maxnode = 0;
    // // int minnode = 999;
    // printf("CarState_NodeID\n");
    // for (i = 0; i < NUMCAR; i++)
    // { // 车辆状态变量：CarState_NodeID、CarState_Time、CarState_CostToNode
    //     printf("%d ", CarState_NodeID[i]);
    //     printf("\n");
    //     // if (maxnode < CarState_NodeID[i])
    //     //     maxnode = CarState_NodeID[i];
    //     // if (minnode > CarState_NodeID[i])
    //     //     minnode = CarState_NodeID[i];
    // }
    // // printf("max node: %d", maxnode);
    // // printf("min node: %d", minnode);

    // 检查初始车辆随机生成地点
    for (i = 0; i < NUMCAR; i++)
    {
        printf("CAR%d's Location: %d\n", i, CarState_NodeID[i]);
    }
}

#endif