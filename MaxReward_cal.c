#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/*
Created on Thursday March 30 2023
@author: lyj

统计调度系统在接受率为100%时，系统获得乘客支付的总收益
结果：679315.812500    68万
first 500:  104855.312500  10.5万
*/

/*定义各类常量*/
// #define NUMREQ 48490
#define NUMREQ_MAX 100000
#define NUMNODE 880 /*gird站点数量，列数*/
#define FEE 3       /*单位为【元/km】，乘客上车之后的收费*/
#define SPEED 0.25  /* 速度，单位为【km/min】，即15km/h*/
// Request
int ReqList_ReqTime[NUMREQ_MAX]; /*请求到达时间*/
int ReqList_UpNode[NUMREQ_MAX];  /*请求对应的上车地点*/
int ReqList_OffNode[NUMREQ_MAX]; /*请求对应的下车地点*/
int ReqList_Test[100000000];
// Network
float MinCost[NUMNODE][NUMNODE]; /*各站点间最小出行时间,单位为min*/
int Nodes_Manhattan[225];        /*曼哈顿研究区域内站点ID*/
// Other
float Max_SysIncreRevd; /*系统累计车费收入*/

int ReqNum_Total; // 一天请求总数

/*Function Prototype*/
void reqdata_input();
void mincost_input();
void reqdata_generation();
void nodes_in_manhattan_fixed();

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
}

void reqdata_generation()
{
    int i, j;
    int num, time;
    // 2、12小时时长的请求数据随机生成，获得该12小时内请求总数
    //  请求三种信息清零
    for (i = 0; i < NUMREQ_MAX; i++) // 12小时请求总数未知，因此先初始化一半，大约25000
    {
        ReqList_ReqTime[i] = 999999;
        ReqList_UpNode[i] = -1;
        ReqList_OffNode[i] = -1;
    }
    // 随机生成100天请求到达时间+数量
    ReqNum_Total = 0;
    for (time = 0; time < 1440; time++)
    {
        if ((time < 9 * 60 && time > 7 * 60) || (time < 19 * 60 && time > 17 * 60))
        {
            num = rand() % 41 + 60; // 高峰期间请求到达频率：60 ～ 100
            for (i = 0; i < num; i++)
            {
                ReqList_ReqTime[ReqNum_Total] = time;
                ReqNum_Total++;
            }
        }
        else if ((time < 13 * 60 && time > 7 * 60) || (time < 23 * 60 && time > 15 * 60))
        {
            num = rand() % 30 + 30; // 次高峰期间请求到达频率：30 ～ 59
            for (i = 0; i < num; i++)
            {
                ReqList_ReqTime[ReqNum_Total] = time;
                ReqNum_Total++;
            }
        }
        else if (time < 6 * 60 && time > 0) // 不包括0分钟
        {
            num = rand() % 6; // 夜间请求到达频率：0 ～ 5
            for (i = 0; i < num; i++)
            {
                ReqList_ReqTime[ReqNum_Total] = time;
                ReqNum_Total++;
            }
        }
        else
        {
            num = rand() % 24 + 6; // 平期间请求到达频率：6 ～ 29
            for (i = 0; i < num; i++)
            {
                ReqList_ReqTime[ReqNum_Total] = time;
                ReqNum_Total++;
            }
        }
    }
    // 随机生成上下车位置
    for (i = 0; i < ReqNum_Total; i++)
    {
        ReqList_UpNode[i] = Nodes_Manhattan[rand() % 225];
        ReqList_OffNode[i] = Nodes_Manhattan[rand() % 225];
        while (ReqList_OffNode[i] == ReqList_UpNode[i]) // 避免上下车站点相同
            ReqList_OffNode[i] = Nodes_Manhattan[rand() % 225];
    }
}

int main()
{
    int req;
    // reqdata_input();
    mincost_input();
    nodes_in_manhattan_fixed();
    reqdata_generation();

    Max_SysIncreRevd = 0;
    for (req = 0; req < ReqNum_Total; req++)
    {
        Max_SysIncreRevd += FEE * SPEED * MinCost[ReqList_UpNode[req]][ReqList_OffNode[req]];
    }
    printf("Max_SysIncreRevd: %f\n", Max_SysIncreRevd);
    printf("ReqNum_Total: %d\n", ReqNum_Total); // 185万

    // printf("%d \n", INT_MAX);
    return 0;
}