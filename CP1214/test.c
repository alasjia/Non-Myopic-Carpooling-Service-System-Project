#include <stdio.h>
#include <stdlib.h>

int main()
{
    int i = 34, j = 65;
    printf("%d\n", i);
    printf("%d\n", j);

    return 0;
}

// 插入排序法：对所有车辆按照距离升序排序
void insert_sort_dis(int cars_after_sort[], int reqid)
{
    int carid, j;                  // 循环变量
    float distance_to_upnode[500]; // 存储到目标站点距离
    // 首先需要按id排序初始化cars_after_sort数组和distance_to_upnode数组
    for (carid = 1; carid < 500; carid++)
    {
        cars_after_sort[carid] = carid;
        distance_to_upnode[carid] = MinCost[CarState_NodeID[carid]][ReqList_UpNode[reqid]];
    }
    // 开始排序
    for (carid = 1; carid < 500; carid++)
    {
        // 用每个车辆id对应的距离作为排序依据
        if (distance_to_upnode[carid] < distance_to_upnode[carid - 1])
        { // 若第 i 个元素大于 i-1 元素则直接插入；反之，需要找到适当的插入位置后在插入。
            j = carid - 1;
            int idx = cars_after_sort[carid];       // 保存哨兵id
            float disx = distance_to_upnode[carid]; // 保存哨兵distance
            while (j > -1 && disx < distance_to_upnode[j])
            { // 采用顺序查找方式找到插入的位置，在查找的同时，将数组中的元素进行后移操作，给插入元素腾出空间
                distance_to_upnode[j + 1] = distance_to_upnode[j];
                cars_after_sort[j + 1] = cars_after_sort[j];
                j--;
            }
            distance_to_upnode[j + 1] = disx;
            cars_after_sort[j + 1] = idx; // 插入到正确位置
        }
        // printf("第%d次排序结果：", i);
        // for (mm = 0; mm < a; mm++)
        //	printf("%d ", carids[mm]);//打印每次排序后的结果
        // printf("\n");
    }
}
