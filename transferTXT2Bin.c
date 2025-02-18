#pragma hdrstop
//---------------------------------------------------------------------------
// #pragma argsused
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

// 计算 rock 中等于 t 的元素数量
int countElements(float*** rock, int lz, int ly, int lx, float t)
{
    int count = 0;
    for (int i = 0; i < lz; i++)
    {
        for (int j = 0; j < ly; j++)
        {
            for (int k = 0; k < lx; k++)
            {
                if (rock[i][j][k] == t)
                {
                    count++;
                }
            }
        }
    }
    return count;
}

int countElements1D(float* rock, int lz, int ly, int lx, float t)
{
    int count = 0;
    int allnumber = lz * ly * lx;
    for (int number = 0; number < allnumber; number++)
    {
        if (rock[number] == t)
        {
            count++;
        }
    }
    return count;
}

// 二进制文件读入
void readCompressedBinary(float* aa, int totalSize, const char* filename)
{

    FILE* file = fopen(filename, "rb");
    if (!file)
    {
        perror("no open file");
        return;
    }

    int buffer = 0;
    int bufferIndex = 0;
    int aaIndex = 0;

    int* A = (int*)calloc(totalSize / 10, sizeof(int));
    int G = 0;
    fread(A, sizeof(int), totalSize / 10, file);

    while (aaIndex < totalSize)
    {
        buffer = A[G];
        G++;
        for (int i = 0; i < 10 && aaIndex < totalSize; i++)
        {

            int shiftAmount = 32 - (i + 1) * 3;
            int value = (buffer >> shiftAmount) & 0x7;

            aa[aaIndex] = (float)value;
            aaIndex++;
        }
    }

    fclose(file);
    free(A);
}


// 将数据写入文件
void write_data_to_file(float*** data, int nz, int ny, int nx, const char* filename)
{
    FILE* file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < nz; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nx; k++)
            {
                fprintf(file, "%d %d %d %f\n", i, j, k, data[i][j][k]);
            }
        }
    }

    fclose(file);
}

// 将数据写入二进制文件
void write_data_to_file2BIN111(float* data, int totalSize, const char* filename)
{
    FILE* file = fopen(filename, "wb");
    if (!file)
    {
        perror("can't open file");
        return;
    }

    int buffer = 0;
    int bufferIndex = 0;
    int G = 0;
    int* A = (int*)calloc(totalSize / 10, sizeof(int));
    if (!A)
    {
        perror("memory allocate error\n");
        fclose(file);
        return;
    }

    for (int i = 0; i < totalSize; i++)
    {
        int value = (int)data[i];

        if (value < 1 || value > 7)
        {
            fprintf(stderr, "The element in the array is out of range (1-7): %d\n", value);
            fclose(file);
            return;
        }
        buffer |= (value << (32 - (bufferIndex + 1) * 3));
        bufferIndex++;

        if (bufferIndex == 10)
        {
            // fwrite(&buffer, sizeof(int), 1, file);
            A[G] = buffer;
            G++;
            buffer = 0;
            bufferIndex = 0;
        }
    }

    fwrite(A, sizeof(int), totalSize / 10, file);

    if (bufferIndex > 0)
    {
        fwrite(&buffer, sizeof(int), 1, file);
    }
    fclose(file);

    free(A);
}

float*** space3d(int nr, int ny, int nc)
{
    float*** a;
    int i, j, k;

    a = (float***)malloc(sizeof(float**) * nr);
    if (a == NULL) {
        printf("内存分配失败: a\n");
        exit(1); // 或者根据需要执行其他错误处理
    }

    for (i = 0; i < nr; i++) {
        a[i] = (float**)malloc(sizeof(float*) * ny);
        if (a[i] == NULL) {
            printf("内存分配失败: a[%d]\n", i);
            exit(1); // 错误处理
        }
    }

    for (i = 0; i < nr; i++) {
        for (j = 0; j < ny; j++) {
            a[i][j] = (float*)malloc(sizeof(float) * nc);
            if (a[i][j] == NULL) {
                printf("内存分配失败: a[%d][%d]\n", i, j);
                exit(1); // 错误处理
            }
        }
    }

    for (i = 0; i < nr; i++)
        for (j = 0; j < ny; j++)
            for (k = 0; k < nc; k++) {
                a[i][j][k] = 0.0f;
            }

    return a;
}


void free_space3d(float*** a, int nr, int ny)
{
    int i, j;
    for (i = 0; i < nr; i++)
    {
        for (j = 0; j < ny; j++)
        {
            free(a[i][j]);
        }
    }
    for (i = 0; i < nr; i++)
    {
        free(a[i]);
    }
    free(a);
}

int main()
{
    int NZ = 1000;
    int NY = 1000;
    int NX = 1000;
    float blockTotalSize = NZ * NY * NX;
    int totalSize = NZ * NY * NX;
    float*** data11;
    data11 = space3d(NZ, NY, NX);
    printf("OPEN FILE\n");
    ////  打开文件
    //clock_t start2, finish2;
    //start2 = clock();
    //FILE* file11;
    //file11 = fopen("H:/超算数字岩心/1000/outputsample1_400x400x400.dat", "r");
    //if (file11 == NULL)
    //{
    //    fprintf(stderr, "无法打开文件\n");
    //    return 1;
    //}
    //// 读取文件内容并存储到数组中
    //int INPUTnz, INPUTny, INPUTnx;
    //float INPUTvalue;
    //while (fscanf(file11, "%d %d %d %f", &INPUTnz, &INPUTny, &INPUTnx, &INPUTvalue) != EOF)
    //{
    //    if (INPUTnz >= 0 && INPUTnz < NZ && INPUTny >= 0 && INPUTny < NY && INPUTnx >= 0 && INPUTnx < NX)
    //    {
    //        data11[INPUTnz][INPUTny][INPUTnx] = INPUTvalue;
    //    }
    //    else
    //    {
    //        fprintf(stderr, "索引超出范围: nz=%d, ny=%d, nx=%d\n", INPUTnz, INPUTny, INPUTnx);
    //    }
    //}
    //// 关闭文件
    //fclose(file11);
    //finish2 = clock();
    //printf("TXT file 读取 %f seconds\n", (float)(finish2 - start2) / CLOCKS_PER_SEC);

    // float PP[7];
    // float PP_sum[7];
    // for (int i = 1; i <= 6; i++)
    // {
    //     int t = i;
    //     int numM = countElements(data11, NZ, NY, NX, t);
    //     PP[i] = (float)numM / blockTotalSize;
    // }
    // for (int i = 1; i <= 6; i++)
    // {
    //    printf("TXT_Component %d ---- %f\n", i, PP[i]);
    // }



    //printf("READ FILE FINISH\n");
    //float* DATA1D = (float*)malloc(NZ * NY * NX * sizeof(float));
    //if (DATA1D == NULL) {
    //    printf("内存分配失败\n");
    //    exit(1); // 或者根据需要执行其他错误处理
    //}
    //else {
    //    printf("内存分配成功\n");
    //}
    //memset(DATA1D, 0, NZ * NY * NX * sizeof(float));
    //for (int z = 0; z < NZ; z++)
    //    for (int y = 0; y < NY; y++)
    //        for (int x = 0; x < NX; x++)
    //        {
    //            DATA1D[z * NX * NY + y * NX + x] = data11[z][y][x];

    //        }

    //float P[7];
    //float P_sum[7];
    //for (int i = 1; i <= 6; i++)
    //{
    //    int t = i;
    //    int num = countElements1D(DATA1D, NZ, NY, NX, t);
    //    P[i] = (float)num / blockTotalSize;
    //}
    //for (int i = 1; i <= 6; i++)
    //{
    //    printf("write BIN_Component %d ---- %f\n", i, P[i]);
    //}

    //// 输出TXT文件
    //char filename[50];
    //sprintf(filename, "1000X1000X1000.txt");
    //clock_t start1, finish1;
    //start1 = clock();
    //write_data_to_file(data11, NZ, NY, NX, filename);
    //finish1 = clock();
    //printf("TXT file 写入 %f seconds\n", (float)(finish1 - start1) / CLOCKS_PER_SEC);

    //// 输出二进制文件
    //char filename11[50];
    //sprintf(filename11, "400X400X400_BIN.bin");
    //clock_t start5, finish5;
    //start5 = clock();
    //write_data_to_file2BIN111(DATA1D, totalSize, filename11);
    //finish5 = clock();
    //printf("Binary file 写入 %f seconds\n", (float)(finish5 - start5) / CLOCKS_PER_SEC);

    float* testre = (float*)malloc(NZ * NY * NX * sizeof(float));
    if (testre == NULL) {
        printf("内存分配失败\n");
        exit(1); // 或者根据需要执行其他错误处理
    }
    else {
        printf("内存分配成功\n");
    }
    memset(testre, 0, NZ * NY * NX * sizeof(float));

    int NNZ = 1000;
    int NNY = 1000;
    int NNX = 1000;

    float* testre1 = (float*)malloc(NNZ * NNY * NNX * sizeof(float));
    if (testre1 == NULL) {
        printf("内存分配失败\n");
        exit(1); // 或者根据需要执行其他错误处理
    }
    else {
        printf("内存分配成功\n");
    }
    memset(testre1, 0, NNZ * NNY * NNX * sizeof(float));

    // 读取二进制文件
    clock_t start3, finish3;
    start3 = clock();
    char* filename00 = "H:/超算数字岩心/1000/1000X1000X1000_BIN.bin";
    FILE* file00 = fopen(filename00, "rb");
    if (file00 == NULL)
    {
        fprintf(stderr, "no open file0\n");
        return 1;
    }
    // 调用函数
    readCompressedBinary(testre, totalSize, filename00);
    finish3 = clock();
    printf("Binary file 读取 %f seconds\n", (float)(finish3 - start3) / CLOCKS_PER_SEC);

    for(int z = 0;z < NNZ; z++)
        for(int y = 0; y < NNY; y++)
            for (int x = 0; x < NNX; x++)
            {
                testre1[z + y * NNX + x * NNX * NNY] = testre[z + y * NNX + x * NNX * NNY];
            }

    float T[7];
    float T_sum[7];
    for (int i = 1; i <= 6; i++)
    {
        int t = i;
        int Tnum = countElements1D(testre1, NNZ, NNY, NNX, t);
        T[i] = (float)Tnum / blockTotalSize;
    }
    for (int i = 1; i <= 6; i++)
    {
        printf("read BIN_Component %d ---- %f\n", i, T[i]);
    }

    int divideX = 10;
    int divideY = 10;
    int divideZ = 1;
    int blockSizeX = NNX / divideX; 
    int blockSizeY = NNY / divideY; 
    int blockSizeZ = NNZ / divideZ; 

    // 开始分块并写入文件
    clock_t start6, finish6;
    start6 = clock();
    for (int ix = 0; ix < divideX; ix++)
    {
        for (int iy = 0; iy < divideY; iy++)
        {
            for (int iz = 0; iz < divideZ; iz++)
            {
                // 计算每个块的总大小
                int blockTotalSize = blockSizeX * blockSizeY * blockSizeZ;
                float* data_local = (float*)malloc(blockTotalSize * sizeof(float));
                if (!data_local)
                {
                    fprintf(stderr, "内存分配失败\n");
                    exit(1);
                }

                // 提取当前块的数据
                int index = 0;
                for (int z = 0; z < blockSizeZ; z++)
                {
                    int globalZ = iz * blockSizeZ + z;
                    for (int y = 0; y < blockSizeY; y++)
                    {
                        int globalY = iy * blockSizeY + y;
                        for (int x = 0; x < blockSizeX; x++)
                        {
                            int globalX = ix * blockSizeX + x;
                            int globalIndex = globalX + NNX * (globalY + NNY * globalZ);
                            data_local[index++] = testre1[globalIndex];
                        }
                    }
                }

                // 创建包含块索引的文件名
                char filename[256];
                sprintf(filename, "%d_%d_%d_%d.bin", NNX, ix, iy, iz);

                // 调用写入函数
                write_data_to_file2BIN111(data_local, blockTotalSize, filename);

                // 释放局部数据
                free(data_local);
            }
        }
    }
    finish6 = clock();
    printf("Binary file 分块写入 %f seconds\n", (float)(finish6 - start6) / CLOCKS_PER_SEC);
    
    // 现在开始读取文件并重建数据
    float* DATA1D_reconstructed = (float*)malloc(NNX * NNY * NNZ * sizeof(float));
    if (DATA1D_reconstructed == NULL)
    {
        fprintf(stderr, "内存分配失败\n");
        exit(1);
    }

    int all = NNX * NNY * NNZ;

    // 初始化重建的数据数组
    for (int i = 0; i < all; i++)
    {
        DATA1D_reconstructed[i] = 0.0f;
    }

    // 读取每个文件并将数据放入重建的数组中
    clock_t start8, finish8;
    start8 = clock();
    for (int ix = 0; ix < divideX; ix++)
    {
        for (int iy = 0; iy < divideY; iy++)
        {
            for (int iz = 0; iz < divideZ; iz++)
            {
                // 计算每个块的总大小
                int blockTotalSize = blockSizeX * blockSizeY * blockSizeZ;
                float* data_local = (float*)malloc(blockTotalSize * sizeof(float));
                if (!data_local)
                {
                    fprintf(stderr, "内存分配失败\n");
                    exit(1);
                }

                // 创建包含块索引的文件名
                char filename[256];
                sprintf(filename, "H:/transferpro/%d_%d_%d_%d.bin", NNX, ix, iy, iz);

                // 调用读取函数
                readCompressedBinary(data_local, blockTotalSize, filename);

                // 将数据放回到重建的数组中
                int index = 0;
                for (int z = 0; z < blockSizeZ; z++)
                {
                    int globalZ = iz * blockSizeZ + z;
                    for (int y = 0; y < blockSizeY; y++)
                    {
                        int globalY = iy * blockSizeY + y;
                        for (int x = 0; x < blockSizeX; x++)
                        {
                            int globalX = ix * blockSizeX + x;
                            int globalIndex = globalX + NNX * (globalY + NNY * globalZ);
                            DATA1D_reconstructed[globalIndex] = data_local[index++];
                        }
                    }
                }

                // 释放局部数据
                free(data_local);
            }
        }
    }
    finish8 = clock();
    printf("Binary file 分块读取 %f seconds\n", (float)(finish8 - start8) / CLOCKS_PER_SEC);
    float TT[8];
    float TT_sum[8];
    for (int i = 1; i <= 7; i++)
    {
        int t = i;
        int TTnum = countElements1D(DATA1D_reconstructed, NNZ, NNY, NNX, t);
        TT[i] = (float)TTnum / blockTotalSize;
        printf("component read BIN_Component %d ---- %f\n", i, TT[i]);
    }



    //float*** data11;
    //data11 = space3d(NZ, NY, NX);
    //for (int z = 0; z < NZ; z++)
    //{
    //    for (int y = 0; y < NY; y++)
    //    {
    //        for (int x = 0; x < NX; x++)
    //        {
    //            data11[z][y][x] = testre[z * NX * NY + y * NX + x];
    //        }
    //    }
    //}
    //// 输出TXT文件
    //char filename11[256];
    //sprintf(filename11, "500X500X500.txt");
    //clock_t start1, finish1;
    //start1 = clock();
    //write_data_to_file(data11, NZ, NY, NX, filename11);
    //finish1 = clock();
    //printf("TXT file write %f seconds\n", (float)(finish1 - start1) / CLOCKS_PER_SEC);

    // 比较原始数据和重建的数据，打印不为零的差异
    int differences = 0;
    for (int i = 0; i < all; i++)
    {
        if (testre1[i] != DATA1D_reconstructed[i])
        {
            differences++;
            printf("Difference at index %d: original = %f, reconstructed = %f\n", i, testre1[i], DATA1D_reconstructed[i]);
            // 为了避免过多输出，可以限制打印的差异数量
            if (differences >= 100)
            {
                printf("超过 100 处差异，停止打印。\n");
                break;
            }
        }
    }

    if (differences == 0)
    {
        printf("数据重建成功，未发现差异。\n");
    }
    else
    {
        printf("数据重建中发现 %d 处差异。\n", differences);
    }



    free(DATA1D_reconstructed);
    free_space3d(data11, NZ, NY);
    //free(DATA1D);
    free(testre);
    free(testre1);
    return 0;
}
