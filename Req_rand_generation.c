#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/*
Created on Friday March 31 2023
@author: lyj

随机生成100天请求数据：到达时间+OD
*/

/*定义各类常量*/
#define NUMNODE 880         /*gird站点数量，列数*/
#define NUMREQ_MAX 10000000 /*目标生成请求数量*/
// Request
int ReqList_ReqTime[NUMREQ_MAX]; /*请求到达时间*/
int ReqList_UpNode[NUMREQ_MAX];  /*请求对应的上车地点*/
int ReqList_OffNode[NUMREQ_MAX]; /*请求对应的下车地点*/
// Network
float MinCost[NUMNODE][NUMNODE]; /*各站点间最小出行时间,单位为min*/
int Nodes_Manhattan[225];        /*曼哈顿研究区域内站点ID*/
// Other

void nodes_in_manhattan_fixed();
void data_write(int req_num);

/*将曼哈顿内站点ID的csv文件输入数组，获得曼哈顿研究区域内站点ID*/
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

void data_write(int req_num)
{
    const char *filename = "RandomRequests.csv"; // 设置文件放置位置
    int i;
    /* File pointer to hold reference to our file */
    FILE *fptr = fopen(filename, "w+"); // fptr为文件指针，可通过fptr来实现对 文件的操作
    /* fopen() return NULL if last operation was unsuccessful */
    if (fptr != NULL)
    { //  若打开文件失败，fopen会返回NULL
        /* Write data to file */
        fprintf(fptr, "%s,%s,%s\n", "ReqTime", "UpNode", "OffNode");
        for (i = 0; i < req_num; i++)
            fprintf(fptr, "%d,%d,%d\n", ReqList_ReqTime[i], ReqList_UpNode[i], ReqList_OffNode[i]);
        // fputs(ReqList_UpNode, fptr);         //fputs()只能写入字符
        /* Close file to save file data */
        fclose(fptr);
        /* Success message */
        printf("Data saved successfully.\n");
    }
}

int main()
{
    int i, j;
    int time, num, total_num, last_ttnum;
    nodes_in_manhattan_fixed();
    // 随机生成100天请求到达时间+数量
    total_num = 0;
    for (j = 0; j < 100; j++)
    {
        last_ttnum = total_num;
        for (time = 0; time < 1440; time++)
        {
            if ((time < 9 * 60 && time > 7 * 60) || (time < 19 * 60 && time > 17 * 60))
            {
                num = rand() % 41 + 60; // 高峰期间请求到达频率：60 ～ 100
                for (i = 0; i < num; i++)
                {
                    ReqList_ReqTime[total_num] = time;
                    total_num++;
                }
            }
            else if ((time < 13 * 60 && time > 7 * 60) || (time < 23 * 60 && time > 15 * 60))
            {
                num = rand() % 30 + 30; // 次高峰期间请求到达频率：30 ～ 59
                for (i = 0; i < num; i++)
                {
                    ReqList_ReqTime[total_num] = time;
                    total_num++;
                }
            }
            else if (time < 6 * 60 && time > 0)
            {
                num = rand() % 6; // 夜间请求到达频率：0 ～ 5
                for (i = 0; i < num; i++)
                {
                    ReqList_ReqTime[total_num] = time;
                    total_num++;
                }
            }
            else
            {
                num = rand() % 24 + 6; // 平期间请求到达频率：6 ～ 29
                for (i = 0; i < num; i++)
                {
                    ReqList_ReqTime[total_num] = time;
                    total_num++;
                }
            }
        }
        printf("request number of day%d: %d \n", j + 1, total_num - last_ttnum);
    }
    // 随机生成上下车位置
    for (i = 0; i < total_num; i++)
    {
        ReqList_UpNode[i] = Nodes_Manhattan[rand() % 225];
        ReqList_OffNode[i] = Nodes_Manhattan[rand() % 225];
        while (ReqList_OffNode[i] == ReqList_UpNode[i])
            ReqList_OffNode[i] = Nodes_Manhattan[rand() % 225];
    }
    data_write(total_num);
    printf("total_num: %d\n", total_num);
    return 0;
}