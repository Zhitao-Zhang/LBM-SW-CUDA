#include <stdio.h>
#include <math.h>
#include <string.h>
#include <slave.h>
#include <simd.h>

#define npop 19
extern int nxt, nyt, nzt;
extern int nxt2, nyt2, nzt2;
extern float G1, G2, B;
extern int Step4D;
extern int myid;
extern int back, forward, left, right;
extern float *obst;
extern int Boundary;
extern float *rho, *rhoexp, *ff, *feq, *tempt;
extern float *ux, *uy, *uz;
extern float *u_x, *u_y, *u_z;
extern float *F1sx, *F1sy, *F1sz, *Fmx1, *Fmy1, *Fmz1, *Fmx2, *Fmy2, *Fmz2;
extern float *fftemp;
extern float *fftemp2;
extern float *pp, *psx, *p;
extern float *psxpretemp, *spretemp;
extern float *psxpretemp2, *spretemp2;
extern float *psxpre, *spre;
extern float *smx_pre1, *smy_pre1, *smz_pre1;

__thread_local int ix, iy, iz;
volatile __thread_local int get_reply, put_reply;

void CalTempt()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    // 分配动态内存
    float ffSlave[slaveAll], temptSlave[slaveAll], obstSlave[slaveAll];
    memset(ffSlave, 0, slaveAll * sizeof(float));
    memset(obstSlave, 0, slaveAll * sizeof(float));
    memset(temptSlave, 0, slaveAll * sizeof(float));
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);

                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                // -----------------------------------------开始计算4D--------------------------------------------------

                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                    get_reply = 0;
                                    athread_get(PE_MODE, &tempt[transferinder4D], &temptSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &obst[transferinder3D], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 3)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        if (obstSlave[SlaveIndex] != 0.0f)
                                        {
                                            temptSlave[SlaveIndex] = ffSlave[SlaveIndex];
                                        }
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &temptSlave[getindex], &tempt[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            // -----------------------------------------开始计算4D--------------------------------------------------

                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                get_reply = 0;
                                athread_get(PE_MODE, &tempt[transferinder4D], &temptSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &obst[transferinder3D], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 3)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {
                                for (ix = 0; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    if (obstSlave[SlaveIndex] != 0.0f)
                                    {
                                        temptSlave[SlaveIndex] = ffSlave[SlaveIndex];
                                    }
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &temptSlave[getindex], &tempt[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 1)
                                    ;
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            // -----------------------------------------开始计算4D--------------------------------------------------

                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                get_reply = 0;
                                athread_get(PE_MODE, &tempt[transferinder4D], &temptSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &obst[transferinder3D], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 3)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < TranferPartX; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    if (obstSlave[SlaveIndex] != 0.0f)
                                    {
                                        temptSlave[SlaveIndex] = ffSlave[SlaveIndex];
                                    }
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &temptSlave[getindex], &tempt[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                while (put_reply != 1)
                                    ;
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        // -----------------------------------------开始计算4D--------------------------------------------------

                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;

                            int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);

                            get_reply = 0;
                            athread_get(PE_MODE, &tempt[transferinder4D], &temptSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &obst[transferinder3D], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 3)
                                ;
                        }
                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {
                            for (ix = 0; ix < SlaveNxt; ix++)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                if (obstSlave[SlaveIndex] != 0.0f)
                                {
                                    temptSlave[SlaveIndex] = ffSlave[SlaveIndex];
                                }
                            }
                        }
                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int getindex = conveyy * SlaveNxt;
                            put_reply = 0;
                            athread_put(PE_MODE, &temptSlave[getindex], &tempt[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            while (put_reply != 1)
                                ;
                        }
                    }
                }
            }
        }
    }
}

void CalBGK()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 10;
    SlaveNyt = 10;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    float tau = 1.0f; // Relaxtion parameter
    // 分配动态内存
    float rhoSlave[slaveAll];
    memset(rhoSlave, 0, slaveAll * sizeof(float));
    float u_xSlave[slaveAll], u_ySlave[slaveAll], u_zSlave[slaveAll], uxSlave[slaveAll], uySlave[slaveAll], uzSlave[slaveAll];
    memset(u_xSlave, 0, slaveAll * sizeof(float));
    memset(u_ySlave, 0, slaveAll * sizeof(float));
    memset(u_zSlave, 0, slaveAll * sizeof(float));
    memset(uxSlave, 0, slaveAll * sizeof(float));
    memset(uySlave, 0, slaveAll * sizeof(float));
    memset(uzSlave, 0, slaveAll * sizeof(float));
    float F1sxSlave[slaveAll], F1sySlave[slaveAll], F1szSlave[slaveAll], Fmx1Slave[slaveAll], Fmy1Slave[slaveAll], Fmz1Slave[slaveAll], Fmx2Slave[slaveAll], Fmy2Slave[slaveAll], Fmz2Slave[slaveAll];
    memset(F1sxSlave, 0, slaveAll * sizeof(float));
    memset(F1sySlave, 0, slaveAll * sizeof(float));
    memset(F1szSlave, 0, slaveAll * sizeof(float));
    memset(Fmx1Slave, 0, slaveAll * sizeof(float));
    memset(Fmy1Slave, 0, slaveAll * sizeof(float));
    memset(Fmz1Slave, 0, slaveAll * sizeof(float));
    memset(Fmx2Slave, 0, slaveAll * sizeof(float));
    memset(Fmy2Slave, 0, slaveAll * sizeof(float));
    memset(Fmz2Slave, 0, slaveAll * sizeof(float));
    // SIMD
    floatv4 u_xSlaveV4, u_ySlaveV4, u_zSlaveV4;
    floatv4 uxSlaveV4, uySlaveV4, uzSlaveV4;
    floatv4 rhoSlaveV4;
    floatv4 F1sxSlaveV4, F1sySlaveV4, F1szSlaveV4, Fmx1SlaveV4, Fmy1SlaveV4, Fmz1SlaveV4, Fmx2SlaveV4, Fmy2SlaveV4, Fmz2SlaveV4;
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;
                                    get_reply = 0;
                                    athread_get(PE_MODE, &ux[transferinder], &uxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &uy[transferinder], &uySlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &uz[transferinder], &uzSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &u_x[transferinder], &u_xSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &u_y[transferinder], &u_ySlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &u_z[transferinder], &u_zSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &F1sx[transferinder], &F1sxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &F1sy[transferinder], &F1sySlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &F1sz[transferinder], &F1szSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 16)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {

                                    for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        simd_loadu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                        simd_loadu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                        simd_loadu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                                        simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                        simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                        simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                        simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                        simd_loadu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                        simd_loadu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                        simd_loadu(F1szSlaveV4, &F1szSlave[SlaveIndex]);
                                        simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                        simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                        simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                        simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                        simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                        simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                        u_xSlaveV4 = uxSlaveV4 + tau * (F1sxSlaveV4 + Fmx1SlaveV4 + Fmx2SlaveV4) / rhoSlaveV4;
                                        u_ySlaveV4 = uySlaveV4 + tau * (F1sySlaveV4 + Fmy1SlaveV4 + Fmy2SlaveV4) / rhoSlaveV4;
                                        u_zSlaveV4 = uzSlaveV4 + tau * (F1szSlaveV4 + Fmz1SlaveV4 + Fmz2SlaveV4) / rhoSlaveV4;

                                        simd_storeu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                        simd_storeu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                        simd_storeu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                                    }
                                    for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                    {

                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        u_xSlave[SlaveIndex] = uxSlave[SlaveIndex] + tau * (F1sxSlave[SlaveIndex] + Fmx1Slave[SlaveIndex] + Fmx2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                                        u_ySlave[SlaveIndex] = uySlave[SlaveIndex] + tau * (F1sySlave[SlaveIndex] + Fmy1Slave[SlaveIndex] + Fmy2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                                        u_zSlave[SlaveIndex] = uzSlave[SlaveIndex] + tau * (F1szSlave[SlaveIndex] + Fmz1Slave[SlaveIndex] + Fmz2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &u_xSlave[getindex], &u_x[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &u_ySlave[getindex], &u_y[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &u_zSlave[getindex], &u_z[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 3)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;
                                get_reply = 0;
                                athread_get(PE_MODE, &ux[transferinder], &uxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &uy[transferinder], &uySlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &uz[transferinder], &uzSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_x[transferinder], &u_xSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_y[transferinder], &u_ySlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_z[transferinder], &u_zSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &F1sx[transferinder], &F1sxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &F1sy[transferinder], &F1sySlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &F1sz[transferinder], &F1szSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 16)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {

                                for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    simd_loadu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                    simd_loadu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                    simd_loadu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                                    simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                    simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                    simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                    simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                    simd_loadu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                    simd_loadu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                    simd_loadu(F1szSlaveV4, &F1szSlave[SlaveIndex]);
                                    simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                    simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                    simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                    simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                    simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                    simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                    u_xSlaveV4 = uxSlaveV4 + tau * (F1sxSlaveV4 + Fmx1SlaveV4 + Fmx2SlaveV4) / rhoSlaveV4;
                                    u_ySlaveV4 = uySlaveV4 + tau * (F1sySlaveV4 + Fmy1SlaveV4 + Fmy2SlaveV4) / rhoSlaveV4;
                                    u_zSlaveV4 = uzSlaveV4 + tau * (F1szSlaveV4 + Fmz1SlaveV4 + Fmz2SlaveV4) / rhoSlaveV4;

                                    simd_storeu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                    simd_storeu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                    simd_storeu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                                }
                                for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                                {

                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    u_xSlave[SlaveIndex] = uxSlave[SlaveIndex] + tau * (F1sxSlave[SlaveIndex] + Fmx1Slave[SlaveIndex] + Fmx2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                                    u_ySlave[SlaveIndex] = uySlave[SlaveIndex] + tau * (F1sySlave[SlaveIndex] + Fmy1Slave[SlaveIndex] + Fmy2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                                    u_zSlave[SlaveIndex] = uzSlave[SlaveIndex] + tau * (F1szSlave[SlaveIndex] + Fmz1Slave[SlaveIndex] + Fmz2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &u_xSlave[getindex], &u_x[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &u_ySlave[getindex], &u_y[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &u_zSlave[getindex], &u_z[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 3)
                                    ;
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy = slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;
                                get_reply = 0;
                                athread_get(PE_MODE, &ux[transferinder], &uxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &uy[transferinder], &uySlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &uz[transferinder], &uzSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_x[transferinder], &u_xSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_y[transferinder], &u_ySlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_z[transferinder], &u_zSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &F1sx[transferinder], &F1sxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &F1sy[transferinder], &F1sySlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &F1sz[transferinder], &F1szSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 16)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {

                                for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    simd_loadu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                    simd_loadu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                    simd_loadu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                                    simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                    simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                    simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                    simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                    simd_loadu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                    simd_loadu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                    simd_loadu(F1szSlaveV4, &F1szSlave[SlaveIndex]);
                                    simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                    simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                    simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                    simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                    simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                    simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                    u_xSlaveV4 = uxSlaveV4 + tau * (F1sxSlaveV4 + Fmx1SlaveV4 + Fmx2SlaveV4) / rhoSlaveV4;
                                    u_ySlaveV4 = uySlaveV4 + tau * (F1sySlaveV4 + Fmy1SlaveV4 + Fmy2SlaveV4) / rhoSlaveV4;
                                    u_zSlaveV4 = uzSlaveV4 + tau * (F1szSlaveV4 + Fmz1SlaveV4 + Fmz2SlaveV4) / rhoSlaveV4;

                                    simd_storeu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                    simd_storeu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                    simd_storeu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                                }
                                for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                {

                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    u_xSlave[SlaveIndex] = uxSlave[SlaveIndex] + tau * (F1sxSlave[SlaveIndex] + Fmx1Slave[SlaveIndex] + Fmx2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                                    u_ySlave[SlaveIndex] = uySlave[SlaveIndex] + tau * (F1sySlave[SlaveIndex] + Fmy1Slave[SlaveIndex] + Fmy2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                                    u_zSlave[SlaveIndex] = uzSlave[SlaveIndex] + tau * (F1szSlave[SlaveIndex] + Fmz1Slave[SlaveIndex] + Fmz2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &u_xSlave[getindex], &u_x[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &u_ySlave[getindex], &u_y[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &u_zSlave[getindex], &u_z[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                while (put_reply != 3)
                                    ;
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;
                            get_reply = 0;
                            athread_get(PE_MODE, &ux[transferinder], &uxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &uy[transferinder], &uySlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &uz[transferinder], &uzSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &u_x[transferinder], &u_xSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &u_y[transferinder], &u_ySlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &u_z[transferinder], &u_zSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &F1sx[transferinder], &F1sxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &F1sy[transferinder], &F1sySlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &F1sz[transferinder], &F1szSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 16)
                                ;
                        }
                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {

                            for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                simd_loadu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                simd_loadu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                simd_loadu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                                simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                simd_loadu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                simd_loadu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                simd_loadu(F1szSlaveV4, &F1szSlave[SlaveIndex]);
                                simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                u_xSlaveV4 = uxSlaveV4 + tau * (F1sxSlaveV4 + Fmx1SlaveV4 + Fmx2SlaveV4) / rhoSlaveV4;
                                u_ySlaveV4 = uySlaveV4 + tau * (F1sySlaveV4 + Fmy1SlaveV4 + Fmy2SlaveV4) / rhoSlaveV4;
                                u_zSlaveV4 = uzSlaveV4 + tau * (F1szSlaveV4 + Fmz1SlaveV4 + Fmz2SlaveV4) / rhoSlaveV4;

                                simd_storeu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                simd_storeu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                simd_storeu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                            }
                            for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                            {

                                int SlaveIndex = iy * SlaveNxt + ix;
                                u_xSlave[SlaveIndex] = uxSlave[SlaveIndex] + tau * (F1sxSlave[SlaveIndex] + Fmx1Slave[SlaveIndex] + Fmx2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                                u_ySlave[SlaveIndex] = uySlave[SlaveIndex] + tau * (F1sySlave[SlaveIndex] + Fmy1Slave[SlaveIndex] + Fmy2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                                u_zSlave[SlaveIndex] = uzSlave[SlaveIndex] + tau * (F1szSlave[SlaveIndex] + Fmz1Slave[SlaveIndex] + Fmz2Slave[SlaveIndex]) / rhoSlave[SlaveIndex];
                            }
                        }
                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int getindex = conveyy * SlaveNxt;
                            put_reply = 0;
                            athread_put(PE_MODE, &u_xSlave[getindex], &u_x[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &u_ySlave[getindex], &u_y[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &u_zSlave[getindex], &u_z[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            while (put_reply != 3)
                                ;
                        }
                    }
                }
            }
        }
    }
}

void CalBGK1()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    float tau = 1.0f; // Relaxtion parameter
    // 权重因子
    int cx1[npop] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 0, 0, 0, 0};
    int cy1[npop] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
    int cz1[npop] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
    float cx1V4 = cx1[Step4D];
    float cy1V4 = cy1[Step4D];
    float cz1V4 = cz1[Step4D];

    float weA1 = 1.0 / 3.0;
    float weA2 = 1.0 / 18.0;
    float weA3 = 1.0 / 36.0;
    float weight[npop] = {weA1, weA2, weA2, weA2, weA2, weA2, weA2, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3};
    float weightV4 = weight[Step4D];
    // 分配动态内存
    float ffSlave[slaveAll], ff1Slave[slaveAll], temptSlave[slaveAll], rhoSlave[slaveAll], feqSlave[slaveAll];
    memset(ffSlave, 0, slaveAll * sizeof(float));
    memset(ff1Slave, 0, slaveAll * sizeof(float));
    memset(temptSlave, 0, slaveAll * sizeof(float));
    memset(rhoSlave, 0, slaveAll * sizeof(float));
    memset(feqSlave, 0, slaveAll * sizeof(float));
    float u_xSlave[slaveAll], u_ySlave[slaveAll], u_zSlave[slaveAll], uxSlave[slaveAll], uySlave[slaveAll], uzSlave[slaveAll], cuSlave[slaveAll];
    memset(u_xSlave, 0, slaveAll * sizeof(float));
    memset(u_ySlave, 0, slaveAll * sizeof(float));
    memset(u_zSlave, 0, slaveAll * sizeof(float));
    memset(uxSlave, 0, slaveAll * sizeof(float));
    memset(uySlave, 0, slaveAll * sizeof(float));
    memset(uzSlave, 0, slaveAll * sizeof(float));
    memset(cuSlave, 0, slaveAll * sizeof(float));
    floatv4 ffSlaveV4, ff1SlaveV4, temptSlaveV4, rhoSlaveV4, feqSlaveV4;
    floatv4 u_xSlaveV4, u_ySlaveV4, u_zSlaveV4, uxSlaveV4, uySlaveV4, uzSlaveV4, cuSlaveV4;
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                // -----------------------------------------开始计算4D--------------------------------------------------
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                    get_reply = 0;
                                    athread_get(PE_MODE, &u_x[transferinder3D], &u_xSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &u_y[transferinder3D], &u_ySlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &u_z[transferinder3D], &u_zSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &feq[transferinder4D], &feqSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &rho[transferinder3D], &rhoSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);

                                    while (get_reply != 6)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;

                                        simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);
                                        simd_loadu(ff1SlaveV4, &ff1Slave[SlaveIndex]);
                                        simd_loadu(temptSlaveV4, &temptSlave[SlaveIndex]);
                                        simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                        simd_loadu(feqSlaveV4, &feqSlave[SlaveIndex]);
                                        simd_loadu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                        simd_loadu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                        simd_loadu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                                        simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                        simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                        simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                        simd_loadu(cuSlaveV4, &cuSlave[SlaveIndex]);
                                        cuSlaveV4 = 3.0f * (cx1V4 * u_xSlaveV4 + cy1V4 * u_ySlaveV4 + cz1V4 * u_zSlaveV4);
                                        feqSlaveV4 = rhoSlaveV4 * weightV4 * (1.0f + cuSlaveV4 + 1.0f / 2.0f * (cuSlaveV4 * cuSlaveV4) - 3.0f / 2.0f * (u_xSlaveV4 * u_xSlaveV4 + u_ySlaveV4 * u_ySlaveV4 + u_zSlaveV4 * u_zSlaveV4));
                                        ff1SlaveV4 = ffSlaveV4 - 1.0f / tau * (ffSlaveV4 - feqSlaveV4);
                                        simd_storeu(feqSlaveV4, &feqSlave[SlaveIndex]);
                                        simd_storeu(ff1SlaveV4, &ff1Slave[SlaveIndex]);
                                    }
                                    for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        cuSlave[SlaveIndex] = 3.0 * (cx1[Step4D] * u_xSlave[SlaveIndex] + cy1[Step4D] * u_ySlave[SlaveIndex] + cz1[Step4D] * u_zSlave[SlaveIndex]);
                                        feqSlave[SlaveIndex] = rhoSlave[SlaveIndex] * weight[Step4D] * (1.0 + cuSlave[SlaveIndex] + 1.0 / 2.0 * (cuSlave[SlaveIndex] * cuSlave[SlaveIndex]) - 3.0 / 2.0 * (u_xSlave[SlaveIndex] * u_xSlave[SlaveIndex] + u_ySlave[SlaveIndex] * u_ySlave[SlaveIndex] + u_zSlave[SlaveIndex] * u_zSlave[SlaveIndex]));
                                        ff1Slave[SlaveIndex] = ffSlave[SlaveIndex] - 1.0 / tau * (ffSlave[SlaveIndex] - feqSlave[SlaveIndex]);
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &feqSlave[getindex], &feq[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &ff1Slave[getindex], &ff[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            // -----------------------------------------开始计算4D--------------------------------------------------
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                get_reply = 0;
                                athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_x[transferinder3D], &u_xSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_y[transferinder3D], &u_ySlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_z[transferinder3D], &u_zSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &feq[transferinder4D], &feqSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &rho[transferinder3D], &rhoSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 6)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {
                                for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;

                                    simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);
                                    simd_loadu(ff1SlaveV4, &ff1Slave[SlaveIndex]);
                                    simd_loadu(temptSlaveV4, &temptSlave[SlaveIndex]);
                                    simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                    simd_loadu(feqSlaveV4, &feqSlave[SlaveIndex]);
                                    simd_loadu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                    simd_loadu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                    simd_loadu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                                    simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                    simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                    simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                    simd_loadu(cuSlaveV4, &cuSlave[SlaveIndex]);
                                    cuSlaveV4 = 3.0f * (cx1V4 * u_xSlaveV4 + cy1V4 * u_ySlaveV4 + cz1V4 * u_zSlaveV4);
                                    feqSlaveV4 = rhoSlaveV4 * weightV4 * (1.0f + cuSlaveV4 + 1.0f / 2.0f * (cuSlaveV4 * cuSlaveV4) - 3.0f / 2.0f * (u_xSlaveV4 * u_xSlaveV4 + u_ySlaveV4 * u_ySlaveV4 + u_zSlaveV4 * u_zSlaveV4));
                                    ff1SlaveV4 = ffSlaveV4 - 1.0f / tau * (ffSlaveV4 - feqSlaveV4);
                                    simd_storeu(feqSlaveV4, &feqSlave[SlaveIndex]);
                                    simd_storeu(ff1SlaveV4, &ff1Slave[SlaveIndex]);
                                }
                                for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    cuSlave[SlaveIndex] = 3.0 * (cx1[Step4D] * u_xSlave[SlaveIndex] + cy1[Step4D] * u_ySlave[SlaveIndex] + cz1[Step4D] * u_zSlave[SlaveIndex]);
                                    feqSlave[SlaveIndex] = rhoSlave[SlaveIndex] * weight[Step4D] * (1.0 + cuSlave[SlaveIndex] + 1.0 / 2.0 * (cuSlave[SlaveIndex] * cuSlave[SlaveIndex]) - 3.0 / 2.0 * (u_xSlave[SlaveIndex] * u_xSlave[SlaveIndex] + u_ySlave[SlaveIndex] * u_ySlave[SlaveIndex] + u_zSlave[SlaveIndex] * u_zSlave[SlaveIndex]));
                                    ff1Slave[SlaveIndex] = ffSlave[SlaveIndex] - 1.0 / tau * (ffSlave[SlaveIndex] - feqSlave[SlaveIndex]);
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &feqSlave[getindex], &feq[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &ff1Slave[getindex], &ff[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 2)
                                    ;
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            // -----------------------------------------开始计算4D--------------------------------------------------
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                get_reply = 0;
                                athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_x[transferinder3D], &u_xSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_y[transferinder3D], &u_ySlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &u_z[transferinder3D], &u_zSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &feq[transferinder4D], &feqSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &rho[transferinder3D], &rhoSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 6)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;

                                    simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);
                                    simd_loadu(ff1SlaveV4, &ff1Slave[SlaveIndex]);
                                    simd_loadu(temptSlaveV4, &temptSlave[SlaveIndex]);
                                    simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                    simd_loadu(feqSlaveV4, &feqSlave[SlaveIndex]);
                                    simd_loadu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                    simd_loadu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                    simd_loadu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                                    simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                    simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                    simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                    simd_loadu(cuSlaveV4, &cuSlave[SlaveIndex]);
                                    cuSlaveV4 = 3.0f * (cx1V4 * u_xSlaveV4 + cy1V4 * u_ySlaveV4 + cz1V4 * u_zSlaveV4);
                                    feqSlaveV4 = rhoSlaveV4 * weightV4 * (1.0f + cuSlaveV4 + 1.0f / 2.0f * (cuSlaveV4 * cuSlaveV4) - 3.0f / 2.0f * (u_xSlaveV4 * u_xSlaveV4 + u_ySlaveV4 * u_ySlaveV4 + u_zSlaveV4 * u_zSlaveV4));
                                    ff1SlaveV4 = ffSlaveV4 - 1.0f / tau * (ffSlaveV4 - feqSlaveV4);
                                    simd_storeu(feqSlaveV4, &feqSlave[SlaveIndex]);
                                    simd_storeu(ff1SlaveV4, &ff1Slave[SlaveIndex]);
                                }
                                for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    cuSlave[SlaveIndex] = 3.0 * (cx1[Step4D] * u_xSlave[SlaveIndex] + cy1[Step4D] * u_ySlave[SlaveIndex] + cz1[Step4D] * u_zSlave[SlaveIndex]);
                                    feqSlave[SlaveIndex] = rhoSlave[SlaveIndex] * weight[Step4D] * (1.0 + cuSlave[SlaveIndex] + 1.0 / 2.0 * (cuSlave[SlaveIndex] * cuSlave[SlaveIndex]) - 3.0 / 2.0 * (u_xSlave[SlaveIndex] * u_xSlave[SlaveIndex] + u_ySlave[SlaveIndex] * u_ySlave[SlaveIndex] + u_zSlave[SlaveIndex] * u_zSlave[SlaveIndex]));
                                    ff1Slave[SlaveIndex] = ffSlave[SlaveIndex] - 1.0 / tau * (ffSlave[SlaveIndex] - feqSlave[SlaveIndex]);
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &feqSlave[getindex], &feq[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &ff1Slave[getindex], &ff[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                while (put_reply != 2)
                                    ;
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        // -----------------------------------------开始计算4D--------------------------------------------------
                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;

                            int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                            get_reply = 0;
                            athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &u_x[transferinder3D], &u_xSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &u_y[transferinder3D], &u_ySlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &u_z[transferinder3D], &u_zSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &feq[transferinder4D], &feqSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &rho[transferinder3D], &rhoSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 6)
                                ;
                        }
                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {
                            for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;

                                simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);
                                simd_loadu(ff1SlaveV4, &ff1Slave[SlaveIndex]);
                                simd_loadu(temptSlaveV4, &temptSlave[SlaveIndex]);
                                simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                simd_loadu(feqSlaveV4, &feqSlave[SlaveIndex]);
                                simd_loadu(u_xSlaveV4, &u_xSlave[SlaveIndex]);
                                simd_loadu(u_ySlaveV4, &u_ySlave[SlaveIndex]);
                                simd_loadu(u_zSlaveV4, &u_zSlave[SlaveIndex]);
                                simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                simd_loadu(cuSlaveV4, &cuSlave[SlaveIndex]);
                                cuSlaveV4 = 3.0f * (cx1V4 * u_xSlaveV4 + cy1V4 * u_ySlaveV4 + cz1V4 * u_zSlaveV4);
                                feqSlaveV4 = rhoSlaveV4 * weightV4 * (1.0f + cuSlaveV4 + 1.0f / 2.0f * (cuSlaveV4 * cuSlaveV4) - 3.0f / 2.0f * (u_xSlaveV4 * u_xSlaveV4 + u_ySlaveV4 * u_ySlaveV4 + u_zSlaveV4 * u_zSlaveV4));
                                ff1SlaveV4 = ffSlaveV4 - 1.0f / tau * (ffSlaveV4 - feqSlaveV4);
                                simd_storeu(feqSlaveV4, &feqSlave[SlaveIndex]);
                                simd_storeu(ff1SlaveV4, &ff1Slave[SlaveIndex]);
                            }
                            for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                cuSlave[SlaveIndex] = 3.0 * (cx1[Step4D] * u_xSlave[SlaveIndex] + cy1[Step4D] * u_ySlave[SlaveIndex] + cz1[Step4D] * u_zSlave[SlaveIndex]);
                                feqSlave[SlaveIndex] = rhoSlave[SlaveIndex] * weight[Step4D] * (1.0 + cuSlave[SlaveIndex] + 1.0 / 2.0 * (cuSlave[SlaveIndex] * cuSlave[SlaveIndex]) - 3.0 / 2.0 * (u_xSlave[SlaveIndex] * u_xSlave[SlaveIndex] + u_ySlave[SlaveIndex] * u_ySlave[SlaveIndex] + u_zSlave[SlaveIndex] * u_zSlave[SlaveIndex]));
                                ff1Slave[SlaveIndex] = ffSlave[SlaveIndex] - 1.0 / tau * (ffSlave[SlaveIndex] - feqSlave[SlaveIndex]);
                            }
                        }
                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int getindex = conveyy * SlaveNxt;
                            put_reply = 0;
                            athread_put(PE_MODE, &feqSlave[getindex], &feq[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &ff1Slave[getindex], &ff[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            while (put_reply != 2)
                                ;
                        }
                    }
                }
            }
        }
    }
}

void UpdateBoundary()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    int opp[npop] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
    // 分配动态内存
    float ffSlave[slaveAll], temptSlave[slaveAll], obstSlave[slaveAll];
    memset(ffSlave, 0, slaveAll * sizeof(float));
    memset(temptSlave, 0, slaveAll * sizeof(float));
    memset(obstSlave, 0, slaveAll * sizeof(float));
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                // -----------------------------------------开始计算4D--------------------------------------------------
                for (int Step4D = 0; Step4D < npop; Step4D++)
                {
                    int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                    int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                    if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                    {
                        if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                        {
                            if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                            {
                                if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)
                                {
                                    //*************************************************************************************************
                                    // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                    //*************************************************************************************************
                                    //-----------------------------------------
                                    // 取数组
                                    //-----------------------------------------
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                        // 本层取值
                                        // 传输数据DMA对应在模型上的坐标索引
                                        // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                        // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                        // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                        // (StepSlaveX) 到索引所在的核组的首地址
                                        // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                        // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                        // (conveyy * nxt) 到索引所在的列了
                                        int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                        int transferinder0 = (opp[Step4D] * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                        int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                        // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                        int getindex = conveyy * SlaveNxt;

                                        get_reply = 0;
                                        athread_get(PE_MODE, &obst[transferinder3D], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                        athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                        athread_get(PE_MODE, &tempt[transferinder0], &temptSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                        while (get_reply != 3)
                                            ;
                                    }
                                    //-----------------------------------------
                                    // 计算
                                    //-----------------------------------------
                                    for (iy = 0; iy < TranferParty; iy++)
                                    {
                                        for (ix = 0; ix < TranferPartX; ix++)
                                        {
                                            int SlaveIndex = iy * SlaveNxt + ix;
                                            if (obstSlave[SlaveIndex] != 0.0f)
                                            {
                                                ffSlave[SlaveIndex] = temptSlave[SlaveIndex];
                                            }
                                        }
                                    }
                                    //-----------------------------------------
                                    // 回传
                                    //-----------------------------------------
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &ffSlave[getindex], &ff[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                            {
                                //*************************************************************************************************
                                // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int transferinder0 = (opp[Step4D] * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &obst[transferinder3D], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &tempt[transferinder0], &temptSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                    while (get_reply != 3)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < SlaveNxt; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        if (obstSlave[SlaveIndex] != 0.0f)
                                        {
                                            ffSlave[SlaveIndex] = temptSlave[SlaveIndex];
                                        }
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &ffSlave[getindex], &ff[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                    }
                    if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)   y 传到SlaveNyt
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int transferinder0 = (opp[Step4D] * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &obst[transferinder3D], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &tempt[transferinder0], &temptSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 3)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < SlaveNyt; iy++)
                                {
                                    for (ix = 0; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        if (obstSlave[SlaveIndex] != 0.0f)
                                        {
                                            ffSlave[SlaveIndex] = temptSlave[SlaveIndex];
                                        }
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &ffSlave[getindex], &ff[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int transferinder0 = (opp[Step4D] * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int transferinder3D = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &obst[transferinder3D], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tempt[transferinder0], &temptSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 3)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    if (obstSlave[SlaveIndex] != 0.0f)
                                    {
                                        ffSlave[SlaveIndex] = temptSlave[SlaveIndex];
                                    }
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &ffSlave[getindex], &ff[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 1)
                                    ;
                            }
                        }
                    }
                }
            }
        }
    }
}

void UpdateMidff()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;

    int cycX, cycY;           // 第几次循环
    int limitnxt, limitnyt;   // 第几个核中(从0开始)
    int justmentX, justmentY; // 判断条件
    // limitnxt = ((nxt % (SlaveNxt * slaveXnumb)) / SlaveNxt) - 1;
    // limitnyt = ((nyt % (SlaveNyt * slaveYnumb)) / SlaveNyt) - 1;

    // cycX = ceil((nxt / (SlaveNxt * slaveXnumb))) - 1;
    // cycY = ceil((nyt / (SlaveNyt * slaveYnumb))) - 1;
    // int justmentY = cycY * (SlaveNyt * slaveYnumb);
    limitnxt = ((int)(((double)(nxt % (SlaveNxt * slaveXnumb)) / SlaveNxt) - 1));
    limitnyt = ((int)(((double)(nyt % (SlaveNyt * slaveYnumb)) / SlaveNyt) - 1));

    cycX = (int)(ceil((double)nxt / (SlaveNxt * slaveXnumb)) - 1);
    cycY = (int)(ceil((double)nyt / (SlaveNyt * slaveYnumb)) - 1);

    justmentX = cycX * (SlaveNxt * slaveXnumb);
    justmentY = cycY * (SlaveNyt * slaveYnumb);

    int cx[npop] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 0, 0, 0, 0};
    int cy[npop] = {0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1};
    int cz[npop] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // 分配动态内存
    float fftempSlave[slaveAll], ffSlave[slaveAll];
    memset(fftempSlave, 0, slaveAll * sizeof(float));
    memset(ffSlave, 0, slaveAll * sizeof(float));
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了

                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);

                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &fftemp[transferinder4D], &fftempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 2)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < TranferPartX; ix++)
                                    {

                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        fftempSlave[SlaveIndex] = ffSlave[SlaveIndex];
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (cx[Step4D] == 1)
                                {
                                    if (StepSlaveX == justmentX && _COL == limitnxt)
                                    {
                                        // x = nxt这一列只传SlaveNxt - 1个数
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                                if (cx[Step4D] == -1)
                                {
                                    if (StepSlaveX == 0 && _COL == 0)
                                    {
                                        // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D] + 1);
                                            int getindex = conveyy * SlaveNxt + 1;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveX == 0 && _COL == 0))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                                if (cy[Step4D] == 1)
                                {
                                    if (StepSlaveY == justmentY && _ROW == limitnyt)
                                    {
                                        // y = nyt这一行不传
                                        for (int conveyy = 0; conveyy < TranferParty - 1; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                    {
                                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                                if (cy[Step4D] == -1)
                                {
                                    if (StepSlaveY == 0 && _ROW == 0)
                                    {
                                        // y = 0这整个一行不传
                                        for (int conveyy = 1; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveY == 0 && _ROW == 0))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                                if (cz[Step4D] == 1)
                                {
                                    if (StepSlaveZ < nzt - 1)
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                                if (cz[Step4D] == -1)
                                {
                                    if (StepSlaveZ > 0)
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了

                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &fftemp[transferinder4D], &fftempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 2)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {
                                for (ix = 0; ix < SlaveNxt; ix++)
                                {

                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    fftempSlave[SlaveIndex] = ffSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                            {
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (cx[Step4D] == 1)
                            {
                                if (StepSlaveX == justmentX && _COL == limitnxt)
                                {
                                    // x = nxt这一列只传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cx[Step4D] == -1)
                            {
                                if (StepSlaveX == 0 && _COL == 0)
                                {
                                    // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D] + 1);
                                        int getindex = conveyy * SlaveNxt + 1;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == 0 && _COL == 0))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == 1)
                            {
                                if (StepSlaveY == justmentY && _ROW == limitnyt)
                                {
                                    // y = nyt这一行不传
                                    for (int conveyy = 0; conveyy < TranferParty - 1; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == -1)
                            {
                                if (StepSlaveY == 0 && _ROW == 0)
                                {
                                    // y = 0这整个一行不传
                                    for (int conveyy = 1; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == 0 && _ROW == 0))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == 1)
                            {
                                if (StepSlaveZ < nzt - 1)
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == -1)
                            {
                                if (StepSlaveZ > 0)
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了

                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &fftemp[transferinder4D], &fftempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 2)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < TranferPartX; ix++)
                                {

                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    fftempSlave[SlaveIndex] = ffSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (cx[Step4D] == 1)
                            {
                                if (StepSlaveX == justmentX && _COL == limitnxt)
                                {
                                    // x = nxt这一列只传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cx[Step4D] == -1)
                            {
                                if (StepSlaveX == 0 && _COL == 0)
                                {
                                    // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D] + 1);
                                        int getindex = conveyy * SlaveNxt + 1;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == 0 && _COL == 0))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == 1)
                            {
                                if (StepSlaveY == justmentY && _ROW == limitnyt)
                                {
                                    // y = nyt这一行不传
                                    for (int conveyy = 0; conveyy < SlaveNyt - 1; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == -1)
                            {
                                if (StepSlaveY == 0 && _ROW == 0)
                                {
                                    // y = 0这整个一行不传
                                    for (int conveyy = 1; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == 0 && _ROW == 0))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == 1)
                            {
                                if (StepSlaveZ < nzt - 1)
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == -1)
                            {
                                if (StepSlaveZ > 0)
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (TranferPartX), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了

                            int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);

                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;

                            get_reply = 0;
                            athread_get(PE_MODE, &fftemp[transferinder4D], &fftempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 2)
                                ;
                        }
                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {
                            for (ix = 0; ix < SlaveNxt; ix++)
                            {

                                int SlaveIndex = iy * SlaveNxt + ix;
                                fftempSlave[SlaveIndex] = ffSlave[SlaveIndex];
                            }
                        }
                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                        {
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                while (put_reply != 1)
                                    ;
                            }
                        }
                        if (cx[Step4D] == 1)
                        {
                            if (StepSlaveX == justmentX && _COL == limitnxt)
                            {
                                // x = nxt这一列只传SlaveNxt - 1个数
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (!(StepSlaveX == justmentX && _COL == limitnxt))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if (cx[Step4D] == -1)
                        {
                            if (StepSlaveX == 0 && _COL == 0)
                            {
                                // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D] + 1);
                                    int getindex = conveyy * SlaveNxt + 1;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (!(StepSlaveX == 0 && _COL == 0))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if (cy[Step4D] == 1)
                        {
                            if (StepSlaveY == justmentY && _ROW == limitnyt)
                            {
                                // y = nyt这一行不传
                                for (int conveyy = 0; conveyy < SlaveNyt - 1; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if (cy[Step4D] == -1)
                        {
                            if (StepSlaveY == 0 && _ROW == 0)
                            {
                                // y = 0这整个一行不传
                                for (int conveyy = 1; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (!(StepSlaveY == 0 && _ROW == 0))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if (cz[Step4D] == 1)
                        {
                            if (StepSlaveZ < nzt - 1)
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if (cz[Step4D] == -1)
                        {
                            if (StepSlaveZ > 0)
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp[transferindertemp], sizeof(float) * (SlaveNxt), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void UpdateMidff1()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;

    int cycX, cycY;           // 第几次循环
    int limitnxt, limitnyt;   // 第几个核中(从0开始)
    int justmentX, justmentY; // 判断条件
    // limitnxt = ((nxt % (SlaveNxt * slaveXnumb)) / SlaveNxt) - 1;
    // limitnyt = ((nyt % (SlaveNyt * slaveYnumb)) / SlaveNyt) - 1;

    // cycX = ceil((nxt / (SlaveNxt * slaveXnumb))) - 1;
    // cycY = ceil((nyt / (SlaveNyt * slaveYnumb))) - 1;
    // int justmentY = cycY * (SlaveNyt * slaveYnumb);
    limitnxt = ((int)(((double)(nxt % (SlaveNxt * slaveXnumb)) / SlaveNxt) - 1));
    limitnyt = ((int)(((double)(nyt % (SlaveNyt * slaveYnumb)) / SlaveNyt) - 1));

    cycX = (int)(ceil((double)nxt / (SlaveNxt * slaveXnumb)) - 1);
    cycY = (int)(ceil((double)nyt / (SlaveNyt * slaveYnumb)) - 1);

    justmentX = cycX * (SlaveNxt * slaveXnumb);
    justmentY = cycY * (SlaveNyt * slaveYnumb);

    int cx[npop] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int cy[npop] = {0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    int cz[npop] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};

    // 分配动态内存
    float fftempSlave[slaveAll], ffSlave[slaveAll];
    memset(fftempSlave, 0, slaveAll * sizeof(float));
    memset(ffSlave, 0, slaveAll * sizeof(float));
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了

                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);

                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &fftemp2[transferinder4D], &fftempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &fftemp[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 2)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < TranferPartX; ix++)
                                    {

                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        fftempSlave[SlaveIndex] = ffSlave[SlaveIndex];
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                                {
                                    // 全传
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (cx[Step4D] == 1)
                                {
                                    if (StepSlaveX == justmentX && _COL == limitnxt)
                                    {
                                        // x = nxt这一列只传SlaveNxt - 1个数
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                                if (cx[Step4D] == -1)
                                {
                                    if (StepSlaveX == 0 && _COL == 0)
                                    {
                                        // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D] + 1);
                                            int getindex = conveyy * SlaveNxt + 1;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveX == 0 && _COL == 0))
                                    {
                                        // 全传
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                                if (cy[Step4D] == 1)
                                {
                                    if (StepSlaveY == justmentY && _ROW == limitnyt)
                                    {
                                        // y = nyt这一行不传
                                        for (int conveyy = 0; conveyy < TranferParty - 1; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                    {
                                        // 全传
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                                if (cy[Step4D] == -1)
                                {
                                    if (StepSlaveY == 0 && _ROW == 0)
                                    {
                                        // y = 0这整个一行不传
                                        for (int conveyy = 1; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveY == 0 && _ROW == 0))
                                    {
                                        // 全传
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                                if (cz[Step4D] == 1)
                                {
                                    if (StepSlaveZ < nzt - 1)
                                    {
                                        // 全传
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                                if (cz[Step4D] == -1)
                                {
                                    if (StepSlaveZ > 0)
                                    {
                                        // 全传
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 1)
                                                ;
                                        }
                                    }
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了

                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &fftemp2[transferinder4D], &fftempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &fftemp[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 2)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {
                                for (ix = 0; ix < SlaveNxt; ix++)
                                {

                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    fftempSlave[SlaveIndex] = ffSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                            {
                                // 全传
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (cx[Step4D] == 1)
                            {
                                if (StepSlaveX == justmentX && _COL == limitnxt)
                                {
                                    // x = nxt这一列只传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cx[Step4D] == -1)
                            {
                                if (StepSlaveX == 0 && _COL == 0)
                                {
                                    // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D] + 1);
                                        int getindex = conveyy * SlaveNxt + 1;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == 0 && _COL == 0))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == 1)
                            {
                                if (StepSlaveY == justmentY && _ROW == limitnyt)
                                {
                                    // y = nyt这一行不传
                                    for (int conveyy = 0; conveyy < TranferParty - 1; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == -1)
                            {
                                if (StepSlaveY == 0 && _ROW == 0)
                                {
                                    // y = 0这整个一行不传
                                    for (int conveyy = 1; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == 0 && _ROW == 0))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == 1)
                            {
                                if (StepSlaveZ < nzt - 1)
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == -1)
                            {
                                if (StepSlaveZ > 0)
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlaveX)   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了

                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &fftemp2[transferinder4D], &fftempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &fftemp[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 2)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < TranferPartX; ix++)
                                {

                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    fftempSlave[SlaveIndex] = ffSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                            {
                                // 全传
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (cx[Step4D] == 1)
                            {
                                if (StepSlaveX == justmentX && _COL == limitnxt)
                                {
                                    // x = nxt这一列只传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                {
                                    // 全传
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cx[Step4D] == -1)
                            {
                                if (StepSlaveX == 0 && _COL == 0)
                                {
                                    // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D] + 1);
                                        int getindex = conveyy * SlaveNxt + 1;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == 0 && _COL == 0))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == 1)
                            {
                                if (StepSlaveY == justmentY && _ROW == limitnyt)
                                {
                                    // y = nyt这一行不传
                                    for (int conveyy = 0; conveyy < SlaveNyt - 1; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == -1)
                            {
                                if (StepSlaveY == 0 && _ROW == 0)
                                {
                                    // y = 0这整个一行不传
                                    for (int conveyy = 1; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == 0 && _ROW == 0))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == 1)
                            {
                                if (StepSlaveZ < nzt - 1)
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == -1)
                            {
                                if (StepSlaveZ > 0)
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 1)
                                            ;
                                    }
                                }
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了

                            int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);

                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;

                            get_reply = 0;
                            athread_get(PE_MODE, &fftemp2[transferinder4D], &fftempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &fftemp[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 2)
                                ;
                        }
                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {
                            for (ix = 0; ix < SlaveNxt; ix++)
                            {

                                int SlaveIndex = iy * SlaveNxt + ix;
                                fftempSlave[SlaveIndex] = ffSlave[SlaveIndex];
                            }
                        }
                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                        {
                            // 全传
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 1)
                                    ;
                            }
                        }
                        if (cx[Step4D] == 1)
                        {
                            if (StepSlaveX == justmentX && _COL == limitnxt)
                            {
                                // x = nxt这一列只传SlaveNxt - 1个数
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (!(StepSlaveX == justmentX && _COL == limitnxt))
                            {
                                // 全传
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if (cx[Step4D] == -1)
                        {
                            if (StepSlaveX == 0 && _COL == 0)
                            {
                                // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D] + 1);
                                    int getindex = conveyy * SlaveNxt + 1;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (!(StepSlaveX == 0 && _COL == 0))
                            {
                                // 全传
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if (cy[Step4D] == 1)
                        {
                            if (StepSlaveY == justmentY && _ROW == limitnyt)
                            {
                                // y = nyt这一行不传
                                for (int conveyy = 0; conveyy < SlaveNyt - 1; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                            {
                                // 全传
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if (cy[Step4D] == -1)
                        {
                            if (StepSlaveY == 0 && _ROW == 0)
                            {
                                // y = 0这整个一行不传
                                for (int conveyy = 1; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                            if (!(StepSlaveY == 0 && _ROW == 0))
                            {
                                // 全传
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if (cz[Step4D] == 1)
                        {
                            if (StepSlaveZ < nzt - 1)
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if (cz[Step4D] == -1)
                        {
                            if (StepSlaveZ > 0)
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ + cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy + cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt + cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &fftempSlave[getindex], &fftemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Updateff()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    // 分配动态内存
    float fftempSlave[slaveAll], ffSlave[slaveAll];
    memset(fftempSlave, 0, slaveAll * sizeof(float));
    memset(ffSlave, 0, slaveAll * sizeof(float));
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                // -----------------------------------------开始计算4D--------------------------------------------------

                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &fftemp2[transferinder4D], &fftempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 2)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        ffSlave[SlaveIndex] = fftempSlave[SlaveIndex];
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &ffSlave[getindex], &ff[transferinder4D], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &fftemp2[transferinder4D], &fftempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 2)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {
                                for (ix = 0; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    ffSlave[SlaveIndex] = fftempSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &ffSlave[getindex], &ff[transferinder4D], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 1)
                                    ;
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &fftemp2[transferinder4D], &fftempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 2)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < TranferPartX; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    ffSlave[SlaveIndex] = fftempSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &ffSlave[getindex], &ff[transferinder4D], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                while (put_reply != 1)
                                    ;
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;

                            get_reply = 0;
                            athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &fftemp2[transferinder4D], &fftempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 2)
                                ;
                        }
                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {
                            for (ix = 0; ix < SlaveNxt; ix++)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                ffSlave[SlaveIndex] = fftempSlave[SlaveIndex];
                            }
                        }
                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int getindex = conveyy * SlaveNxt;
                            put_reply = 0;
                            athread_put(PE_MODE, &ffSlave[getindex], &ff[transferinder4D], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            while (put_reply != 1)
                                ;
                        }
                    }
                }
            }
        }
    }
}

void Updaterho()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    // 分配动态内存
    float rhoSlave[slaveAll], ffSlave[slaveAll];
    memset(rhoSlave, 0, slaveAll * sizeof(float));
    memset(ffSlave, 0, slaveAll * sizeof(float));
    floatv4 rhoSlaveV4, ffSlaveV4;
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                memset(rhoSlave, 0, slaveAll * sizeof(float));
                // -----------------------------------------开始计算4D--------------------------------------------------
                for (int Step4D = 0; Step4D < npop; Step4D++)
                {
                    if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                    {
                        if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                        {
                            if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                            {
                                if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                                {
                                    //*************************************************************************************************
                                    // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                    //*************************************************************************************************
                                    //-----------------------------------------
                                    // 取数组
                                    //-----------------------------------------
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                        // 本层取值
                                        // 传输数据DMA对应在模型上的坐标索引
                                        // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                        // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                        // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                        // (StepSlaveX) 到索引所在的核组的首地址
                                        // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                        // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                        // (conveyy * nxt) 到索引所在的列了
                                        int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                        // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                        int getindex = conveyy * SlaveNxt;

                                        get_reply = 0;
                                        athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                        while (get_reply != 1)
                                            ;
                                    }
                                    //-----------------------------------------
                                    // 计算
                                    //-----------------------------------------
                                    for (iy = 0; iy < TranferParty; iy++)
                                    {
                                        for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                        {
                                            int SlaveIndex = iy * SlaveNxt + ix;

                                            simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                            simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);

                                            rhoSlaveV4 = rhoSlaveV4 + ffSlaveV4;

                                            simd_storeu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                        }
                                        for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                        {
                                            int SlaveIndex = iy * SlaveNxt + ix;
                                            rhoSlave[SlaveIndex] = rhoSlave[SlaveIndex] + ffSlave[SlaveIndex];
                                        }
                                    }
                                }
                            }
                            if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                            {
                                //*************************************************************************************************
                                // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                    while (get_reply != 1)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;

                                        simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                        simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);

                                        rhoSlaveV4 = rhoSlaveV4 + ffSlaveV4;

                                        simd_storeu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                    }
                                    for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        rhoSlave[SlaveIndex] = rhoSlave[SlaveIndex] + ffSlave[SlaveIndex];
                                    }
                                }
                            }
                        }
                    }
                    if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 1)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < SlaveNyt; iy++)
                                {
                                    for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;

                                        simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                        simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);

                                        rhoSlaveV4 = rhoSlaveV4 + ffSlaveV4;

                                        simd_storeu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                    }
                                    for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        rhoSlave[SlaveIndex] = rhoSlave[SlaveIndex] + ffSlave[SlaveIndex];
                                    }
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 1)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;

                                    simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                    simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);

                                    rhoSlaveV4 = rhoSlaveV4 + ffSlaveV4;

                                    simd_storeu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                }
                                for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    rhoSlave[SlaveIndex] = rhoSlave[SlaveIndex] + ffSlave[SlaveIndex];
                                }
                            }
                        }
                    }
                }
                //-----------------------------------------
                // 回传
                //-----------------------------------------
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &rhoSlave[getindex], &rho[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 1)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &rhoSlave[getindex], &rho[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 1)
                                    ;
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &rhoSlave[getindex], &rho[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                while (put_reply != 1)
                                    ;
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int getindex = conveyy * SlaveNxt;
                            put_reply = 0;
                            athread_put(PE_MODE, &rhoSlave[getindex], &rho[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            while (put_reply != 1)
                                ;
                        }
                    }
                }
            }
        }
    }
}

void UpdateUPot()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    int cx1[npop] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 0, 0, 0, 0};
    int cy1[npop] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
    int cz1[npop] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
    float rhoexpSlave;
    // 分配动态内存
    float rhoSlave[slaveAll], ffSlave[slaveAll], uxSlave[slaveAll], uySlave[slaveAll], uzSlave[slaveAll], ppSlave[slaveAll], psxSlave[slaveAll];
    memset(rhoSlave, 0, slaveAll * sizeof(float));
    memset(ffSlave, 0, slaveAll * sizeof(float));
    memset(uxSlave, 0, slaveAll * sizeof(float));
    memset(uySlave, 0, slaveAll * sizeof(float));
    memset(uzSlave, 0, slaveAll * sizeof(float));
    memset(ppSlave, 0, slaveAll * sizeof(float));
    memset(psxSlave, 0, slaveAll * sizeof(float));
    floatv4 uxSlaveV4, uySlaveV4, uzSlaveV4, ffSlaveV4, rhoSlaveV4;
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                memset(uxSlave, 0, slaveAll * sizeof(float));
                memset(uySlave, 0, slaveAll * sizeof(float));
                memset(uzSlave, 0, slaveAll * sizeof(float));
                // -----------------------------------------开始计算4D--------------------------------------------------
                for (int Step4D = 0; Step4D < npop; Step4D++)
                {
                    int cx1V4 = cx1[Step4D];
                    int cy1V4 = cy1[Step4D];
                    int cz1V4 = cz1[Step4D];
                    if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                    {
                        if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                        {
                            if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                            {
                                if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                                {
                                    //*************************************************************************************************
                                    // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                    //*************************************************************************************************
                                    //-----------------------------------------
                                    // 取数组
                                    //-----------------------------------------
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                        // 本层取值
                                        // 传输数据DMA对应在模型上的坐标索引
                                        // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                        // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                        // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                        // (StepSlaveX) 到索引所在的核组的首地址
                                        // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                        // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                        // (conveyy * nxt) 到索引所在的列了
                                        int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                        // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                        int getindex = conveyy * SlaveNxt;

                                        get_reply = 0;
                                        athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                        while (get_reply != 1)
                                            ;
                                    }
                                    //-----------------------------------------
                                    // 计算
                                    //-----------------------------------------
                                    for (iy = 0; iy < TranferParty; iy++)
                                    {
                                        for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                        {
                                            int SlaveIndex = iy * SlaveNxt + ix;
                                            simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                            simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                            simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                            simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);

                                            uxSlaveV4 = uxSlaveV4 + ffSlaveV4 * cx1V4;
                                            uySlaveV4 = uySlaveV4 + ffSlaveV4 * cy1V4;
                                            uzSlaveV4 = uzSlaveV4 + ffSlaveV4 * cz1V4;

                                            simd_storeu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                            simd_storeu(uySlaveV4, &uySlave[SlaveIndex]);
                                            simd_storeu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                        }
                                        for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                        {
                                            int SlaveIndex = iy * SlaveNxt + ix;
                                            uxSlave[SlaveIndex] = uxSlave[SlaveIndex] + ffSlave[SlaveIndex] * cx1[Step4D];
                                            uySlave[SlaveIndex] = uySlave[SlaveIndex] + ffSlave[SlaveIndex] * cy1[Step4D];
                                            uzSlave[SlaveIndex] = uzSlave[SlaveIndex] + ffSlave[SlaveIndex] * cz1[Step4D];
                                        }
                                    }
                                }
                            }
                            if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                            {
                                //*************************************************************************************************
                                // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                    while (get_reply != 1)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                        simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                        simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                        simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);

                                        uxSlaveV4 = uxSlaveV4 + ffSlaveV4 * cx1V4;
                                        uySlaveV4 = uySlaveV4 + ffSlaveV4 * cy1V4;
                                        uzSlaveV4 = uzSlaveV4 + ffSlaveV4 * cz1V4;

                                        simd_storeu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                        simd_storeu(uySlaveV4, &uySlave[SlaveIndex]);
                                        simd_storeu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                    }
                                    for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        uxSlave[SlaveIndex] = uxSlave[SlaveIndex] + ffSlave[SlaveIndex] * cx1[Step4D];
                                        uySlave[SlaveIndex] = uySlave[SlaveIndex] + ffSlave[SlaveIndex] * cy1[Step4D];
                                        uzSlave[SlaveIndex] = uzSlave[SlaveIndex] + ffSlave[SlaveIndex] * cz1[Step4D];
                                    }
                                }
                            }
                        }
                    }
                    if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 1)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < SlaveNyt; iy++)
                                {
                                    for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                        simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                        simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                        simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);

                                        uxSlaveV4 = uxSlaveV4 + ffSlaveV4 * cx1V4;
                                        uySlaveV4 = uySlaveV4 + ffSlaveV4 * cy1V4;
                                        uzSlaveV4 = uzSlaveV4 + ffSlaveV4 * cz1V4;

                                        simd_storeu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                        simd_storeu(uySlaveV4, &uySlave[SlaveIndex]);
                                        simd_storeu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                    }
                                    for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        uxSlave[SlaveIndex] = uxSlave[SlaveIndex] + ffSlave[SlaveIndex] * cx1[Step4D];
                                        uySlave[SlaveIndex] = uySlave[SlaveIndex] + ffSlave[SlaveIndex] * cy1[Step4D];
                                        uzSlave[SlaveIndex] = uzSlave[SlaveIndex] + ffSlave[SlaveIndex] * cz1[Step4D];
                                    }
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &ff[transferinder4D], &ffSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 1)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                    simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                    simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                    simd_loadu(ffSlaveV4, &ffSlave[SlaveIndex]);

                                    uxSlaveV4 = uxSlaveV4 + ffSlaveV4 * cx1V4;
                                    uySlaveV4 = uySlaveV4 + ffSlaveV4 * cy1V4;
                                    uzSlaveV4 = uzSlaveV4 + ffSlaveV4 * cz1V4;

                                    simd_storeu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                    simd_storeu(uySlaveV4, &uySlave[SlaveIndex]);
                                    simd_storeu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                }
                                for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    uxSlave[SlaveIndex] = uxSlave[SlaveIndex] + ffSlave[SlaveIndex] * cx1[Step4D];
                                    uySlave[SlaveIndex] = uySlave[SlaveIndex] + ffSlave[SlaveIndex] * cy1[Step4D];
                                    uzSlave[SlaveIndex] = uzSlave[SlaveIndex] + ffSlave[SlaveIndex] * cz1[Step4D];
                                }
                            }
                        }
                    }
                }

                // -----------------------------------------开始计算3D--------------------------------------------------
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &pp[transferinder], &ppSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 3)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                        simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                        simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                        simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);

                                        uxSlaveV4 = uxSlaveV4 / rhoSlaveV4;
                                        uySlaveV4 = uySlaveV4 / rhoSlaveV4;
                                        uzSlaveV4 = uzSlaveV4 / rhoSlaveV4;

                                        simd_storeu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                        simd_storeu(uySlaveV4, &uySlave[SlaveIndex]);
                                        simd_storeu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                    }
                                    for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        uxSlave[SlaveIndex] = uxSlave[SlaveIndex] / rhoSlave[SlaveIndex];
                                        uySlave[SlaveIndex] = uySlave[SlaveIndex] / rhoSlave[SlaveIndex];
                                        uzSlave[SlaveIndex] = uzSlave[SlaveIndex] / rhoSlave[SlaveIndex];
                                    }
                                    for (ix = 0; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        rhoexpSlave = exp(-rhoSlave[SlaveIndex]);
                                        ppSlave[SlaveIndex] = rhoSlave[SlaveIndex] / 3.0 + 3.0 * G1 * (1.0 - rhoexpSlave) * (1.0 - rhoexpSlave);
                                        psxSlave[SlaveIndex] = sqrt(fabs((ppSlave[SlaveIndex] - rhoSlave[SlaveIndex] / 3.0) / G1) * 6.0);
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &uxSlave[getindex], &ux[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &uySlave[getindex], &uy[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &uzSlave[getindex], &uz[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &ppSlave[getindex], &pp[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxSlave[getindex], &psx[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 5)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                // athread_get(PE_MODE, &rhoexp[transferinder], &rhoexpSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &pp[transferinder], &ppSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 3)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {
                                for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                    simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                    simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                    simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);

                                    uxSlaveV4 = uxSlaveV4 / rhoSlaveV4;
                                    uySlaveV4 = uySlaveV4 / rhoSlaveV4;
                                    uzSlaveV4 = uzSlaveV4 / rhoSlaveV4;

                                    simd_storeu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                    simd_storeu(uySlaveV4, &uySlave[SlaveIndex]);
                                    simd_storeu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                }
                                for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    uxSlave[SlaveIndex] = uxSlave[SlaveIndex] / rhoSlave[SlaveIndex];
                                    uySlave[SlaveIndex] = uySlave[SlaveIndex] / rhoSlave[SlaveIndex];
                                    uzSlave[SlaveIndex] = uzSlave[SlaveIndex] / rhoSlave[SlaveIndex];
                                }
                                for (ix = 0; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    rhoexpSlave = exp(-rhoSlave[SlaveIndex]);
                                    ppSlave[SlaveIndex] = rhoSlave[SlaveIndex] / 3.0 + 3.0 * G1 * (1.0 - rhoexpSlave) * (1.0 - rhoexpSlave);
                                    psxSlave[SlaveIndex] = sqrt(fabs((ppSlave[SlaveIndex] - rhoSlave[SlaveIndex] / 3.0) / G1) * 6.0);
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &uxSlave[getindex], &ux[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &uySlave[getindex], &uy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &uzSlave[getindex], &uz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);

                                athread_put(PE_MODE, &ppSlave[getindex], &pp[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &psxSlave[getindex], &psx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);

                                while (put_reply != 5)
                                    ;
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &pp[transferinder], &ppSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 3)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                    simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                    simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                    simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);

                                    uxSlaveV4 = uxSlaveV4 / rhoSlaveV4;
                                    uySlaveV4 = uySlaveV4 / rhoSlaveV4;
                                    uzSlaveV4 = uzSlaveV4 / rhoSlaveV4;

                                    simd_storeu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                    simd_storeu(uySlaveV4, &uySlave[SlaveIndex]);
                                    simd_storeu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                }
                                for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    uxSlave[SlaveIndex] = uxSlave[SlaveIndex] / rhoSlave[SlaveIndex];
                                    uySlave[SlaveIndex] = uySlave[SlaveIndex] / rhoSlave[SlaveIndex];
                                    uzSlave[SlaveIndex] = uzSlave[SlaveIndex] / rhoSlave[SlaveIndex];
                                }
                                for (ix = 0; ix < TranferPartX; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    rhoexpSlave = exp(-rhoSlave[SlaveIndex]);
                                    ppSlave[SlaveIndex] = rhoSlave[SlaveIndex] / 3.0 + 3.0 * G1 * (1.0 - rhoexpSlave) * (1.0 - rhoexpSlave);
                                    psxSlave[SlaveIndex] = sqrt(fabs((ppSlave[SlaveIndex] - rhoSlave[SlaveIndex] / 3.0) / G1) * 6.0);
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &uxSlave[getindex], &ux[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &uySlave[getindex], &uy[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &uzSlave[getindex], &uz[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &ppSlave[getindex], &pp[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &psxSlave[getindex], &psx[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                while (put_reply != 5)
                                    ;
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;

                            get_reply = 0;
                            athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            // athread_get(PE_MODE, &rhoexp[transferinder], &rhoexpSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pp[transferinder], &ppSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 3)
                                ;
                        }
                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {
                            for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                simd_loadu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                simd_loadu(uySlaveV4, &uySlave[SlaveIndex]);
                                simd_loadu(uzSlaveV4, &uzSlave[SlaveIndex]);
                                simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);

                                uxSlaveV4 = uxSlaveV4 / rhoSlaveV4;
                                uySlaveV4 = uySlaveV4 / rhoSlaveV4;
                                uzSlaveV4 = uzSlaveV4 / rhoSlaveV4;

                                simd_storeu(uxSlaveV4, &uxSlave[SlaveIndex]);
                                simd_storeu(uySlaveV4, &uySlave[SlaveIndex]);
                                simd_storeu(uzSlaveV4, &uzSlave[SlaveIndex]);
                            }
                            for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                uxSlave[SlaveIndex] = uxSlave[SlaveIndex] / rhoSlave[SlaveIndex];
                                uySlave[SlaveIndex] = uySlave[SlaveIndex] / rhoSlave[SlaveIndex];
                                uzSlave[SlaveIndex] = uzSlave[SlaveIndex] / rhoSlave[SlaveIndex];
                            }
                            for (ix = 0; ix < SlaveNxt; ix++)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                rhoexpSlave = exp(-rhoSlave[SlaveIndex]);
                                ppSlave[SlaveIndex] = rhoSlave[SlaveIndex] / 3.0 + 3.0 * G1 * (1.0 - rhoexpSlave) * (1.0 - rhoexpSlave);
                                psxSlave[SlaveIndex] = sqrt(fabs((ppSlave[SlaveIndex] - rhoSlave[SlaveIndex] / 3.0) / G1) * 6.0);
                            }
                        }
                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int getindex = conveyy * SlaveNxt;
                            put_reply = 0;
                            athread_put(PE_MODE, &uxSlave[getindex], &ux[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &uySlave[getindex], &uy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &uzSlave[getindex], &uz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);

                            athread_put(PE_MODE, &ppSlave[getindex], &pp[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &psxSlave[getindex], &psx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);

                            while (put_reply != 5)
                                ;
                        }
                    }
                }
            }
        }
    }
}

void PotBoundaryCalP()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    // 分配动态内存
    float obstSlave[slaveAll], psxSlave[slaveAll], pSlave[slaveAll], rhoSlave[slaveAll];
    memset(obstSlave, 0, slaveAll * sizeof(float));
    memset(psxSlave, 0, slaveAll * sizeof(float));
    memset(pSlave, 0, slaveAll * sizeof(float));
    memset(rhoSlave, 0, slaveAll * sizeof(float));
    floatv4 psxSlaveV4, pSlaveV4, rhoSlaveV4;
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                // -----------------------------------------开始计算3D--------------------------------------------------
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 3)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        if (obstSlave[SlaveIndex] != 0.0f)
                                        {
                                            psxSlave[SlaveIndex] = 0.0f;
                                        }
                                    }
                                    for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;

                                        simd_loadu(pSlaveV4, &pSlave[SlaveIndex]);
                                        simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                        simd_loadu(psxSlaveV4, &psxSlave[SlaveIndex]);
                                        pSlaveV4 = rhoSlaveV4 / 3.0f + G1 / 6.0f * psxSlaveV4 * psxSlaveV4;
                                        simd_storeu(pSlaveV4, &pSlave[SlaveIndex]);
                                        simd_storeu(psxSlaveV4, &psxSlave[SlaveIndex]);
                                    }
                                    for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        pSlave[SlaveIndex] = rhoSlave[SlaveIndex] / 3.0 + G1 / 6.0 * psxSlave[SlaveIndex] * psxSlave[SlaveIndex];
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &psxSlave[getindex], &psx[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &pSlave[getindex], &p[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 3)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {
                                for (ix = 0; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    if (obstSlave[SlaveIndex] != 0.0f)
                                    {
                                        psxSlave[SlaveIndex] = 0.0f;
                                    }
                                }
                                for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;

                                    simd_loadu(pSlaveV4, &pSlave[SlaveIndex]);
                                    simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                    simd_loadu(psxSlaveV4, &psxSlave[SlaveIndex]);
                                    pSlaveV4 = rhoSlaveV4 / 3.0f + G1 / 6.0f * psxSlaveV4 * psxSlaveV4;
                                    simd_storeu(pSlaveV4, &pSlave[SlaveIndex]);
                                    simd_storeu(psxSlaveV4, &psxSlave[SlaveIndex]);
                                }
                                for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    pSlave[SlaveIndex] = rhoSlave[SlaveIndex] / 3.0 + G1 / 6.0 * psxSlave[SlaveIndex] * psxSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &psxSlave[getindex], &psx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pSlave[getindex], &p[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 2)
                                    ;
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 3)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < TranferPartX; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    if (obstSlave[SlaveIndex] != 0.0f)
                                    {
                                        psxSlave[SlaveIndex] = 0.0f;
                                    }
                                }
                                for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;

                                    simd_loadu(pSlaveV4, &pSlave[SlaveIndex]);
                                    simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                    simd_loadu(psxSlaveV4, &psxSlave[SlaveIndex]);
                                    pSlaveV4 = rhoSlaveV4 / 3.0f + G1 / 6.0f * psxSlaveV4 * psxSlaveV4;
                                    simd_storeu(pSlaveV4, &pSlave[SlaveIndex]);
                                    simd_storeu(psxSlaveV4, &psxSlave[SlaveIndex]);
                                }
                                for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    pSlave[SlaveIndex] = rhoSlave[SlaveIndex] / 3.0 + G1 / 6.0 * psxSlave[SlaveIndex] * psxSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &psxSlave[getindex], &psx[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pSlave[getindex], &p[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                while (put_reply != 2)
                                    ;
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;

                            get_reply = 0;
                            athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &rho[transferinder], &rhoSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 3)
                                ;
                        }
                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {
                            for (ix = 0; ix < SlaveNxt; ix++)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                if (obstSlave[SlaveIndex] != 0.0f)
                                {
                                    psxSlave[SlaveIndex] = 0.0f;
                                }
                            }
                            for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;

                                simd_loadu(pSlaveV4, &pSlave[SlaveIndex]);
                                simd_loadu(rhoSlaveV4, &rhoSlave[SlaveIndex]);
                                simd_loadu(psxSlaveV4, &psxSlave[SlaveIndex]);
                                pSlaveV4 = rhoSlaveV4 / 3.0f + G1 / 6.0f * psxSlaveV4 * psxSlaveV4;
                                simd_storeu(pSlaveV4, &pSlave[SlaveIndex]);
                                simd_storeu(psxSlaveV4, &psxSlave[SlaveIndex]);
                            }
                            for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                pSlave[SlaveIndex] = rhoSlave[SlaveIndex] / 3.0 + G1 / 6.0 * psxSlave[SlaveIndex] * psxSlave[SlaveIndex];
                            }
                        }
                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int getindex = conveyy * SlaveNxt;
                            put_reply = 0;
                            athread_put(PE_MODE, &psxSlave[getindex], &psx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &pSlave[getindex], &p[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            while (put_reply != 2)
                                ;
                        }
                    }
                }
            }
        }
    }
}

void UpdateMidPot()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;

    int cycX, cycY;           // 第几次循环
    int limitnxt, limitnyt;   // 第几个核中(从0开始)
    int justmentX, justmentY; // 判断条件
    limitnxt = ((int)(((double)(nxt % (SlaveNxt * slaveXnumb)) / SlaveNxt) - 1));
    limitnyt = ((int)(((double)(nyt % (SlaveNyt * slaveYnumb)) / SlaveNyt) - 1));

    cycX = (int)(ceil((double)nxt / (SlaveNxt * slaveXnumb)) - 1);
    cycY = (int)(ceil((double)nyt / (SlaveNyt * slaveYnumb)) - 1);

    justmentX = cycX * (SlaveNxt * slaveXnumb);
    justmentY = cycY * (SlaveNyt * slaveYnumb);

    int cx[npop] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 0, 0, 0, 0};
    int cy[npop] = {0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1};
    int cz[npop] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // 分配动态内存
    float psxpretempSlave[slaveAll], spretempSlave[slaveAll];
    memset(psxpretempSlave, 0, slaveAll * sizeof(float));
    memset(spretempSlave, 0, slaveAll * sizeof(float));
    float psxSlave[slaveAll], obstSlave[slaveAll];
    memset(psxSlave, 0, slaveAll * sizeof(float));
    memset(obstSlave, 0, slaveAll * sizeof(float));
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &spretemp[transferinder4D], &spretempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &psxpretemp[transferinder4D], &psxpretempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 4)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        spretempSlave[SlaveIndex] = obstSlave[SlaveIndex];
                                        psxpretempSlave[SlaveIndex] = psxSlave[SlaveIndex];
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                                {
                                    // 全传
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (cx[Step4D] == 1)
                                {
                                    if (StepSlaveX == 0 && _COL == 0)
                                    {
                                        // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D] + 1);
                                            int getindex = conveyy * SlaveNxt + 1;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveX == 0 && _COL == 0))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                                if (cx[Step4D] == -1)
                                {
                                    if (StepSlaveX == justmentX && _COL == limitnxt)
                                    {
                                        // x = nxt这一列只传SlaveNxt - 1个数
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                                if (cy[Step4D] == 1)
                                {
                                    if (StepSlaveY == 0 && _ROW == 0)
                                    {
                                        // y = 0这整个一行不传
                                        for (int conveyy = 1; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveY == 0 && _ROW == 0))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                                if (cy[Step4D] == -1)
                                {
                                    if (StepSlaveY == justmentY && _ROW == limitnyt)
                                    {
                                        // y = nyt这一行不传
                                        for (int conveyy = 0; conveyy < TranferParty - 1; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                                if (cz[Step4D] == 1)
                                {
                                    if (StepSlaveZ > 0)
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                                if (cz[Step4D] == -1)
                                {
                                    if (StepSlaveZ < nzt - 1)
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &psxpretemp[transferinder4D], &psxpretempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &spretemp[transferinder4D], &spretempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 4)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {
                                for (ix = 0; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    spretempSlave[SlaveIndex] = obstSlave[SlaveIndex];
                                    psxpretempSlave[SlaveIndex] = psxSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                            {
                                // 全传
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                            if (cx[Step4D] == 1)
                            {
                                if (StepSlaveX == 0 && _COL == 0)
                                {
                                    // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D] + 1);
                                        int getindex = conveyy * SlaveNxt + 1;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == 0 && _COL == 0))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cx[Step4D] == -1)
                            {
                                if (StepSlaveX == justmentX && _COL == limitnxt)
                                {
                                    // x = nxt这一列只传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == 1)
                            {
                                if (StepSlaveY == 0 && _ROW == 0)
                                {
                                    // y = 0这整个一行不传
                                    for (int conveyy = 1; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == 0 && _ROW == 0))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == -1)
                            {
                                if (StepSlaveY == justmentY && _ROW == limitnyt)
                                {
                                    // y = nyt这一行不传
                                    for (int conveyy = 0; conveyy < TranferParty - 1; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == 1)
                            {
                                if (StepSlaveZ > 0)
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == -1)
                            {
                                if (StepSlaveZ < nzt - 1)
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &psxpretemp[transferinder4D], &psxpretempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &spretemp[transferinder4D], &spretempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 4)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < TranferPartX; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    spretempSlave[SlaveIndex] = obstSlave[SlaveIndex];
                                    psxpretempSlave[SlaveIndex] = psxSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                            {
                                // 全传
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                            if (cx[Step4D] == 1)
                            {
                                if (StepSlaveX == 0 && _COL == 0)
                                {
                                    // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D] + 1);
                                        int getindex = conveyy * SlaveNxt + 1;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == 0 && _COL == 0))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cx[Step4D] == -1)
                            {
                                if (StepSlaveX == justmentX && _COL == limitnxt)
                                {
                                    // x = nxt这一列只传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == 1)
                            {
                                if (StepSlaveY == 0 && _ROW == 0)
                                {
                                    // y = 0这整个一行不传
                                    for (int conveyy = 1; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == 0 && _ROW == 0))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == -1)
                            {
                                if (StepSlaveY == justmentY && _ROW == limitnyt)
                                {
                                    // y = nyt这一行不传
                                    for (int conveyy = 0; conveyy < SlaveNyt - 1; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == 1)
                            {
                                if (StepSlaveZ > 0)
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == -1)
                            {
                                if (StepSlaveZ < nzt - 1)
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;

                            get_reply = 0;
                            athread_get(PE_MODE, &psxpretemp[transferinder4D], &psxpretempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &spretemp[transferinder4D], &spretempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 4)
                                ;
                        }
                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {
                            for (ix = 0; ix < SlaveNxt; ix++)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                spretempSlave[SlaveIndex] = obstSlave[SlaveIndex];
                                psxpretempSlave[SlaveIndex] = psxSlave[SlaveIndex];
                            }
                        }
                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                        {
                            // 全传
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 2)
                                    ;
                            }
                        }
                        if (cx[Step4D] == 1)
                        {
                            if (StepSlaveX == 0 && _COL == 0)
                            {
                                // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D] + 1);
                                    int getindex = conveyy * SlaveNxt + 1;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                            if (!(StepSlaveX == 0 && _COL == 0))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if (cx[Step4D] == -1)
                        {
                            if (StepSlaveX == justmentX && _COL == limitnxt)
                            {
                                // x = nxt这一列只传SlaveNxt - 1个数
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                            if (!(StepSlaveX == justmentX && _COL == limitnxt))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if (cy[Step4D] == 1)
                        {
                            if (StepSlaveY == 0 && _ROW == 0)
                            {
                                // y = 0这整个一行不传
                                for (int conveyy = 1; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                            if (!(StepSlaveY == 0 && _ROW == 0))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if (cy[Step4D] == -1)
                        {
                            if (StepSlaveY == justmentY && _ROW == limitnyt)
                            {
                                // y = nyt这一行不传
                                for (int conveyy = 0; conveyy < SlaveNyt - 1; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                            if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if (cz[Step4D] == 1)
                        {
                            if (StepSlaveZ > 0)
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if (cz[Step4D] == -1)
                        {
                            if (StepSlaveZ < nzt - 1)
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void UpdateMidPot1()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    int cycX, cycY;           // 第几次循环
    int limitnxt, limitnyt;   // 第几个核中(从0开始)
    int justmentX, justmentY; // 判断条件

    limitnxt = ((int)(((double)(nxt % (SlaveNxt * slaveXnumb)) / SlaveNxt) - 1));
    limitnyt = ((int)(((double)(nyt % (SlaveNyt * slaveYnumb)) / SlaveNyt) - 1));

    cycX = (int)(ceil((double)nxt / (SlaveNxt * slaveXnumb)) - 1);
    cycY = (int)(ceil((double)nyt / (SlaveNyt * slaveYnumb)) - 1);

    justmentX = cycX * (SlaveNxt * slaveXnumb);
    justmentY = cycY * (SlaveNyt * slaveYnumb);
    // if (_PEN == 5 && Step4D == 10 && myid == 0)
    // {
    //     printf("jusY = %d\n", justmentY);
    //     printf("jusX = %d\n", justmentX);
    //     printf("limitnxt = %d\n", limitnxt);
    //     printf("limitnyt = %d\n", limitnyt);
    //     printf("cycX = %d\n", cycX);
    //     printf("cycY = %d\n", cycY);
    // }

    int cx[npop] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int cy[npop] = {0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    int cz[npop] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};

    // 分配动态内存
    float psxpretempSlave[slaveAll], spretempSlave[slaveAll];
    memset(psxpretempSlave, 0, slaveAll * sizeof(float));
    memset(spretempSlave, 0, slaveAll * sizeof(float));
    float psxSlave[slaveAll], obstSlave[slaveAll];
    memset(psxSlave, 0, slaveAll * sizeof(float));
    memset(obstSlave, 0, slaveAll * sizeof(float));
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &spretemp[transferinder4D], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &psxpretemp[transferinder4D], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    // athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    // athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 2)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        // spretempSlave[SlaveIndex] = _PEN * 100000 + iy * SlaveNxt + ix + 100 * Step4D;
                                        spretempSlave[SlaveIndex] = obstSlave[SlaveIndex];
                                        psxpretempSlave[SlaveIndex] = psxSlave[SlaveIndex];
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                                {
                                    // 全传
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (cx[Step4D] == 1)
                                {
                                    if (StepSlaveX == 0 && _COL == 0)
                                    {
                                        // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D] + 1);
                                            int getindex = conveyy * SlaveNxt + 1;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveX == 0 && _COL == 0))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                                if (cx[Step4D] == -1)
                                {
                                    if (StepSlaveX == justmentX && _COL == limitnxt)
                                    {
                                        // x = nxt这一列只传SlaveNxt - 1个数
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                                if (cy[Step4D] == 1)
                                {
                                    if (StepSlaveY == 0 && _ROW == 0)
                                    {
                                        // y = 0这整个一行不传
                                        for (int conveyy = 1; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveY == 0 && _ROW == 0))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                                if (cy[Step4D] == -1)
                                {
                                    if (StepSlaveY == justmentY && _ROW == limitnyt)
                                    {
                                        // y = nyt这一行不传
                                        for (int conveyy = 0; conveyy < TranferParty - 1; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                    if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                                if (cz[Step4D] == 1)
                                {
                                    if (StepSlaveZ > 0)
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                                if (cz[Step4D] == -1)
                                {
                                    if (StepSlaveZ < nzt - 1)
                                    {
                                        for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                        {
                                            int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                            int getindex = conveyy * SlaveNxt;
                                            put_reply = 0;
                                            athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                            while (put_reply != 2)
                                                ;
                                        }
                                    }
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &psxpretemp[transferinder4D], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &spretemp[transferinder4D], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                // athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                // athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 2)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {
                                for (ix = 0; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    // spretempSlave[SlaveIndex] = _PEN * 100000 + iy * SlaveNxt + ix + 100 * Step4D;
                                    spretempSlave[SlaveIndex] = obstSlave[SlaveIndex];
                                    psxpretempSlave[SlaveIndex] = psxSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                            {
                                // 全传
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                            if (cx[Step4D] == 1)
                            {
                                if (StepSlaveX == 0 && _COL == 0)
                                {
                                    // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D] + 1);
                                        int getindex = conveyy * SlaveNxt + 1;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == 0 && _COL == 0))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cx[Step4D] == -1)
                            {
                                if (StepSlaveX == justmentX && _COL == limitnxt)
                                {
                                    // x = nxt这一列只传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == 1)
                            {
                                if (StepSlaveY == 0 && _ROW == 0)
                                {
                                    // y = 0这整个一行不传
                                    for (int conveyy = 1; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == 0 && _ROW == 0))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == -1)
                            {
                                if (StepSlaveY == justmentY && _ROW == limitnyt)
                                {
                                    // y = nyt这一行不传
                                    for (int conveyy = 0; conveyy < TranferParty - 1; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == 1)
                            {
                                if (StepSlaveZ > 0)
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == -1)
                            {
                                if (StepSlaveZ < nzt - 1)
                                {
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &psxpretemp[transferinder4D], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &spretemp[transferinder4D], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                // athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                // athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 2)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < TranferPartX; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    // spretempSlave[SlaveIndex] = _PEN * 100000 + iy * SlaveNxt + ix + 100 * Step4D;
                                    spretempSlave[SlaveIndex] = obstSlave[SlaveIndex];
                                    psxpretempSlave[SlaveIndex] = psxSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                            {
                                // 全传
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                            if (cx[Step4D] == 1)
                            {
                                if (StepSlaveX == 0 && _COL == 0)
                                {
                                    // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D] + 1);
                                        int getindex = conveyy * SlaveNxt + 1;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == 0 && _COL == 0))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cx[Step4D] == -1)
                            {
                                if (StepSlaveX == justmentX && _COL == limitnxt)
                                {
                                    // x = nxt这一列只传SlaveNxt - 1个数
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * (TranferPartX - 1), &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveX == justmentX && _COL == limitnxt))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == 1)
                            {
                                if (StepSlaveY == 0 && _ROW == 0)
                                {
                                    // y = 0这整个一行不传
                                    for (int conveyy = 1; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == 0 && _ROW == 0))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cy[Step4D] == -1)
                            {
                                if (StepSlaveY == justmentY && _ROW == limitnyt)
                                {
                                    // y = nyt这一行不传
                                    for (int conveyy = 0; conveyy < SlaveNyt - 1; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                                if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == 1)
                            {
                                if (StepSlaveZ > 0)
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if (cz[Step4D] == -1)
                            {
                                if (StepSlaveZ < nzt - 1)
                                {
                                    for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                    {
                                        int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;

                            get_reply = 0;
                            athread_get(PE_MODE, &psxpretemp[transferinder4D], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &spretemp[transferinder4D], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            // athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            // athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 2)
                                ;
                        }
                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {
                            for (ix = 0; ix < SlaveNxt; ix++)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                // spretempSlave[SlaveIndex] = _PEN * 100000 + iy * SlaveNxt + ix + 100 * Step4D;
                                spretempSlave[SlaveIndex] = obstSlave[SlaveIndex];
                                psxpretempSlave[SlaveIndex] = psxSlave[SlaveIndex];
                            }
                        }
                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        if (cx[Step4D] == 0 && cy[Step4D] == 0 && cz[Step4D] == 0)
                        {
                            // 全传
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 2)
                                    ;
                            }
                        }
                        if (cx[Step4D] == 1)
                        {
                            if (StepSlaveX == 0 && _COL == 0)
                            {
                                // x = 0这一列从第二个首地址开始传SlaveNxt - 1个数
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D] + 1);
                                    int getindex = conveyy * SlaveNxt + 1;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                            if (!(StepSlaveX == 0 && _COL == 0))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if (cx[Step4D] == -1)
                        {
                            if (StepSlaveX == justmentX && _COL == limitnxt)
                            {
                                // x = nxt这一列只传SlaveNxt - 1个数
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * (SlaveNxt - 1), &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                    // if (Step4D == 10 && myid == 0 && conveyy == 0 StepSlaveZ == 10 && StepSlaveX == 0 && StepSlaveY == 0)
                                    // {
                                    //     printf("x\n");
                                    // }
                                }
                            }
                            if (!(StepSlaveX == justmentX && _COL == limitnxt))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if (cy[Step4D] == 1)
                        {
                            if (StepSlaveY == 0 && _ROW == 0)
                            {
                                // y = 0这整个一行不传
                                for (int conveyy = 1; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                            if (!(StepSlaveY == 0 && _ROW == 0))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if (cy[Step4D] == -1)
                        {
                            if (StepSlaveY == justmentY && (_ROW == limitnyt))
                            {
                                // y = nyt这一行不传
                                for (int conveyy = 0; conveyy < SlaveNyt - 1; conveyy++)
                                {

                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                    // if (Step4D == 10 && myid == 0 && conveyy == 0 StepSlaveZ == 10 && StepSlaveX == 0 && StepSlaveY == 0)
                                    // {
                                    //     printf("Y\n");
                                    // }
                                }
                            }
                            if (!(StepSlaveY == justmentY && _ROW == limitnyt))
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if (cz[Step4D] == 1)
                        {
                            if (StepSlaveZ > 0)
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if (cz[Step4D] == -1)
                        {
                            if (StepSlaveZ < nzt - 1)
                            {
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferindertemp = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ - cz[Step4D])) + (StepSlaveY + _ROW * SlaveNyt + conveyy - cy[Step4D]) * nxt + (StepSlaveX + _COL * SlaveNxt - cx[Step4D]);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spretempSlave[getindex], &spretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpretempSlave[getindex], &psxpretemp2[transferindertemp], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void UpdatePot()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    int cx[npop] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 0, 0, 0, 0};
    int cy[npop] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
    int cz[npop] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
    // 分配动态内存
    float psxpretempSlave[slaveAll], spretempSlave[slaveAll];
    memset(psxpretempSlave, 0, slaveAll * sizeof(float));
    memset(spretempSlave, 0, slaveAll * sizeof(float));
    float psxpreSlave[slaveAll], spreSlave[slaveAll];
    memset(psxpreSlave, 0, slaveAll * sizeof(float));
    memset(spreSlave, 0, slaveAll * sizeof(float));
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                // -----------------------------------------开始计算4D--------------------------------------------------
                for (int Step4D = 0; Step4D < npop; Step4D++)
                {
                    if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                    {
                        if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                        {
                            if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                            {
                                if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                                {
                                    //*************************************************************************************************
                                    // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                    //*************************************************************************************************
                                    //-----------------------------------------
                                    // 取数组
                                    //-----------------------------------------
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                        // 本层取值
                                        // 传输数据DMA对应在模型上的坐标索引
                                        // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                        // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                        // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                        // (StepSlaveX) 到索引所在的核组的首地址
                                        // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                        // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                        // (conveyy * nxt) 到索引所在的列了
                                        int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                        // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                        int getindex = conveyy * SlaveNxt;

                                        get_reply = 0;
                                        athread_get(PE_MODE, &psxpretemp2[transferinder4D], &psxpretempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                        athread_get(PE_MODE, &spretemp2[transferinder4D], &spretempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                        athread_get(PE_MODE, &spre[transferinder4D], &spreSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                        athread_get(PE_MODE, &psxpre[transferinder4D], &psxpreSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                        while (get_reply != 4)
                                            ;
                                    }
                                    //-----------------------------------------
                                    // 计算
                                    //-----------------------------------------
                                    for (iy = 0; iy < TranferParty; iy++)
                                    {
                                        for (ix = 0; ix < TranferPartX; ix++)
                                        {
                                            int SlaveIndex = iy * SlaveNxt + ix;
                                            psxpreSlave[SlaveIndex] = psxpretempSlave[SlaveIndex];
                                            spreSlave[SlaveIndex] = spretempSlave[SlaveIndex];
                                        }
                                    }
                                    //-----------------------------------------
                                    // 回传
                                    //-----------------------------------------
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                        int getindex = conveyy * SlaveNxt;
                                        put_reply = 0;
                                        athread_put(PE_MODE, &spreSlave[getindex], &spre[transferinder4D], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        athread_put(PE_MODE, &psxpreSlave[getindex], &psxpre[transferinder4D], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                        while (put_reply != 2)
                                            ;
                                    }
                                }
                            }
                            if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                            {
                                //*************************************************************************************************
                                // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &psxpretemp2[transferinder4D], &psxpretempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &spretemp2[transferinder4D], &spretempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &spre[transferinder4D], &spreSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &psxpre[transferinder4D], &psxpreSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                    while (get_reply != 4)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < SlaveNxt; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        psxpreSlave[SlaveIndex] = psxpretempSlave[SlaveIndex];
                                        spreSlave[SlaveIndex] = spretempSlave[SlaveIndex];
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spreSlave[getindex], &spre[transferinder4D], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpreSlave[getindex], &psxpre[transferinder4D], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                    }
                    if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &psxpretemp2[transferinder4D], &psxpretempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &spretemp2[transferinder4D], &spretempSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &spre[transferinder4D], &spreSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &psxpre[transferinder4D], &psxpreSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 4)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < SlaveNyt; iy++)
                                {
                                    for (ix = 0; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        psxpreSlave[SlaveIndex] = psxpretempSlave[SlaveIndex];
                                        spreSlave[SlaveIndex] = spretempSlave[SlaveIndex];
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &spreSlave[getindex], &spre[transferinder4D], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &psxpreSlave[getindex], &psxpre[transferinder4D], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 2)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &psxpretemp2[transferinder4D], &psxpretempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &spretemp2[transferinder4D], &spretempSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &spre[transferinder4D], &spreSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &psxpre[transferinder4D], &psxpreSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 4)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    psxpreSlave[SlaveIndex] = psxpretempSlave[SlaveIndex];
                                    spreSlave[SlaveIndex] = spretempSlave[SlaveIndex];
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &spreSlave[getindex], &spre[transferinder4D], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &psxpreSlave[getindex], &psxpre[transferinder4D], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 2)
                                    ;
                            }
                        }
                    }
                }
            }
        }
    }
}

void CalF()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    float weA3 = 1.0 / 36.0;
    float weA4 = 1.0 / 6.0;
    float weA5 = 1.0 / 12.0;
    int cx1[npop] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 0, 0, 0, 0};
    int cy1[npop] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
    int cz1[npop] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
    float weight1[npop] = {0.0, weA4, weA4, weA4, weA4, weA4, weA4, weA5, weA5, weA5, weA5, weA5, weA5, weA5, weA5, weA5, weA5, weA5, weA5};

    // 分配动态内存
    float smx_pre1Slave[slaveAll], smy_pre1Slave[slaveAll], smz_pre1Slave[slaveAll];
    memset(smx_pre1Slave, 0, slaveAll * sizeof(float));
    memset(smy_pre1Slave, 0, slaveAll * sizeof(float));
    memset(smz_pre1Slave, 0, slaveAll * sizeof(float));
    float Fmx1Slave[slaveAll], Fmy1Slave[slaveAll], Fmz1Slave[slaveAll], Fmx2Slave[slaveAll], Fmy2Slave[slaveAll], Fmz2Slave[slaveAll];
    memset(Fmx1Slave, 0, slaveAll * sizeof(float));
    memset(Fmy1Slave, 0, slaveAll * sizeof(float));
    memset(Fmz1Slave, 0, slaveAll * sizeof(float));
    memset(Fmx2Slave, 0, slaveAll * sizeof(float));
    memset(Fmy2Slave, 0, slaveAll * sizeof(float));
    memset(Fmz2Slave, 0, slaveAll * sizeof(float));
    float spreSlave[slaveAll], psxpreSlave[slaveAll];
    memset(spreSlave, 0, slaveAll * sizeof(float));
    memset(psxpreSlave, 0, slaveAll * sizeof(float));
    floatv4 smx_pre1SlaveV4, smy_pre1SlaveV4, smz_pre1SlaveV4;
    floatv4 Fmx1SlaveV4, Fmy1SlaveV4, Fmz1SlaveV4, Fmx2SlaveV4, Fmy2SlaveV4, Fmz2SlaveV4;
    floatv4 spreSlaveV4, psxpreSlaveV4;
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                memset(smx_pre1Slave, 0, slaveAll * sizeof(float));
                memset(smy_pre1Slave, 0, slaveAll * sizeof(float));
                memset(smz_pre1Slave, 0, slaveAll * sizeof(float));
                memset(Fmx1Slave, 0, slaveAll * sizeof(float));
                memset(Fmy1Slave, 0, slaveAll * sizeof(float));
                memset(Fmz1Slave, 0, slaveAll * sizeof(float));
                memset(Fmx2Slave, 0, slaveAll * sizeof(float));
                memset(Fmy2Slave, 0, slaveAll * sizeof(float));
                memset(Fmz2Slave, 0, slaveAll * sizeof(float));
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                // -----------------------------------------开始计算4D--------------------------------------------------
                for (int Step4D = 0; Step4D < npop; Step4D++)
                {
                    float cx1V4 = cx1[Step4D];
                    float cy1V4 = cy1[Step4D];
                    float cz1V4 = cz1[Step4D];
                    float weight1V4 = weight1[Step4D];
                    if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                    {
                        if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                        {
                            if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                            {
                                if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                                {
                                    //*************************************************************************************************
                                    // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                    //*************************************************************************************************
                                    //-----------------------------------------
                                    // 取数组
                                    //-----------------------------------------
                                    for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                    {
                                        //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                        // 本层取值
                                        // 传输数据DMA对应在模型上的坐标索引
                                        // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                        // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                        // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                        // (StepSlaveX) 到索引所在的核组的首地址
                                        // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                        // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                        // (conveyy * nxt) 到索引所在的列了
                                        int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                        // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                        int getindex = conveyy * SlaveNxt;

                                        get_reply = 0;
                                        athread_get(PE_MODE, &spre[transferinder4D], &spreSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                        athread_get(PE_MODE, &psxpre[transferinder4D], &psxpreSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                        while (get_reply != 2)
                                            ;
                                    }
                                    //-----------------------------------------
                                    // 计算
                                    //-----------------------------------------
                                    for (iy = 0; iy < TranferParty; iy++)
                                    {
                                        for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                        {
                                            int SlaveIndex = iy * SlaveNxt + ix;

                                            simd_loadu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                            simd_loadu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                            simd_loadu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                            simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                            simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                            simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                            simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                            simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                            simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                            simd_loadu(spreSlaveV4, &spreSlave[SlaveIndex]);
                                            simd_loadu(psxpreSlaveV4, &psxpreSlave[SlaveIndex]);
                                            smx_pre1SlaveV4 = smx_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cx1V4;
                                            smy_pre1SlaveV4 = smy_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cy1V4;
                                            smz_pre1SlaveV4 = smz_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cz1V4;
                                            Fmx1SlaveV4 = Fmx1SlaveV4 + weight1V4 * psxpreSlaveV4 * cx1V4;
                                            Fmy1SlaveV4 = Fmy1SlaveV4 + weight1V4 * psxpreSlaveV4 * cy1V4;
                                            Fmz1SlaveV4 = Fmz1SlaveV4 + weight1V4 * psxpreSlaveV4 * cz1V4;
                                            Fmx2SlaveV4 = Fmx2SlaveV4 + weight1V4 * psxpreSlaveV4 * cx1V4 * psxpreSlaveV4;
                                            Fmy2SlaveV4 = Fmy2SlaveV4 + weight1V4 * psxpreSlaveV4 * cy1V4 * psxpreSlaveV4;
                                            Fmz2SlaveV4 = Fmz2SlaveV4 + weight1V4 * psxpreSlaveV4 * cz1V4 * psxpreSlaveV4;

                                            simd_storeu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                            simd_storeu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                            simd_storeu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                            simd_storeu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                            simd_storeu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                            simd_storeu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                            simd_storeu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                            simd_storeu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                            simd_storeu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                        }
                                        for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                        {
                                            int SlaveIndex = iy * SlaveNxt + ix;

                                            smx_pre1Slave[SlaveIndex] = smx_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cx1[Step4D];
                                            smy_pre1Slave[SlaveIndex] = smy_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cy1[Step4D];
                                            smz_pre1Slave[SlaveIndex] = smz_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cz1[Step4D];
                                            Fmx1Slave[SlaveIndex] = Fmx1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cx1[Step4D];
                                            Fmy1Slave[SlaveIndex] = Fmy1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cy1[Step4D];
                                            Fmz1Slave[SlaveIndex] = Fmz1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cz1[Step4D];
                                            Fmx2Slave[SlaveIndex] = Fmx2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cx1[Step4D] * psxpreSlave[SlaveIndex];
                                            Fmy2Slave[SlaveIndex] = Fmy2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cy1[Step4D] * psxpreSlave[SlaveIndex];
                                            Fmz2Slave[SlaveIndex] = Fmz2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cz1[Step4D] * psxpreSlave[SlaveIndex];
                                        }
                                    }
                                }
                            }
                            if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                            {
                                //*************************************************************************************************
                                // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &spre[transferinder4D], &spreSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &psxpre[transferinder4D], &psxpreSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                    while (get_reply != 2)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;

                                        simd_loadu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                        simd_loadu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                        simd_loadu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                        simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                        simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                        simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                        simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                        simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                        simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                        simd_loadu(spreSlaveV4, &spreSlave[SlaveIndex]);
                                        simd_loadu(psxpreSlaveV4, &psxpreSlave[SlaveIndex]);
                                        smx_pre1SlaveV4 = smx_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cx1V4;
                                        smy_pre1SlaveV4 = smy_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cy1V4;
                                        smz_pre1SlaveV4 = smz_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cz1V4;
                                        Fmx1SlaveV4 = Fmx1SlaveV4 + weight1V4 * psxpreSlaveV4 * cx1V4;
                                        Fmy1SlaveV4 = Fmy1SlaveV4 + weight1V4 * psxpreSlaveV4 * cy1V4;
                                        Fmz1SlaveV4 = Fmz1SlaveV4 + weight1V4 * psxpreSlaveV4 * cz1V4;
                                        Fmx2SlaveV4 = Fmx2SlaveV4 + weight1V4 * psxpreSlaveV4 * cx1V4 * psxpreSlaveV4;
                                        Fmy2SlaveV4 = Fmy2SlaveV4 + weight1V4 * psxpreSlaveV4 * cy1V4 * psxpreSlaveV4;
                                        Fmz2SlaveV4 = Fmz2SlaveV4 + weight1V4 * psxpreSlaveV4 * cz1V4 * psxpreSlaveV4;

                                        simd_storeu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                        simd_storeu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                        simd_storeu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                        simd_storeu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                        simd_storeu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                        simd_storeu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                        simd_storeu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                        simd_storeu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                        simd_storeu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                    }
                                    for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;

                                        smx_pre1Slave[SlaveIndex] = smx_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cx1[Step4D];
                                        smy_pre1Slave[SlaveIndex] = smy_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cy1[Step4D];
                                        smz_pre1Slave[SlaveIndex] = smz_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cz1[Step4D];
                                        Fmx1Slave[SlaveIndex] = Fmx1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cx1[Step4D];
                                        Fmy1Slave[SlaveIndex] = Fmy1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cy1[Step4D];
                                        Fmz1Slave[SlaveIndex] = Fmz1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cz1[Step4D];
                                        Fmx2Slave[SlaveIndex] = Fmx2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cx1[Step4D] * psxpreSlave[SlaveIndex];
                                        Fmy2Slave[SlaveIndex] = Fmy2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cy1[Step4D] * psxpreSlave[SlaveIndex];
                                        Fmz2Slave[SlaveIndex] = Fmz2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cz1[Step4D] * psxpreSlave[SlaveIndex];
                                    }
                                }
                            }
                        }
                    }
                    if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &spre[transferinder4D], &spreSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &psxpre[transferinder4D], &psxpreSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 2)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < SlaveNyt; iy++)
                                {
                                    for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;

                                        simd_loadu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                        simd_loadu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                        simd_loadu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                        simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                        simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                        simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                        simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                        simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                        simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                        simd_loadu(spreSlaveV4, &spreSlave[SlaveIndex]);
                                        simd_loadu(psxpreSlaveV4, &psxpreSlave[SlaveIndex]);
                                        smx_pre1SlaveV4 = smx_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cx1V4;
                                        smy_pre1SlaveV4 = smy_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cy1V4;
                                        smz_pre1SlaveV4 = smz_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cz1V4;
                                        Fmx1SlaveV4 = Fmx1SlaveV4 + weight1V4 * psxpreSlaveV4 * cx1V4;
                                        Fmy1SlaveV4 = Fmy1SlaveV4 + weight1V4 * psxpreSlaveV4 * cy1V4;
                                        Fmz1SlaveV4 = Fmz1SlaveV4 + weight1V4 * psxpreSlaveV4 * cz1V4;
                                        Fmx2SlaveV4 = Fmx2SlaveV4 + weight1V4 * psxpreSlaveV4 * cx1V4 * psxpreSlaveV4;
                                        Fmy2SlaveV4 = Fmy2SlaveV4 + weight1V4 * psxpreSlaveV4 * cy1V4 * psxpreSlaveV4;
                                        Fmz2SlaveV4 = Fmz2SlaveV4 + weight1V4 * psxpreSlaveV4 * cz1V4 * psxpreSlaveV4;

                                        simd_storeu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                        simd_storeu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                        simd_storeu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                        simd_storeu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                        simd_storeu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                        simd_storeu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                        simd_storeu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                        simd_storeu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                        simd_storeu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                    }
                                    for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;

                                        smx_pre1Slave[SlaveIndex] = smx_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cx1[Step4D];
                                        smy_pre1Slave[SlaveIndex] = smy_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cy1[Step4D];
                                        smz_pre1Slave[SlaveIndex] = smz_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cz1[Step4D];
                                        Fmx1Slave[SlaveIndex] = Fmx1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cx1[Step4D];
                                        Fmy1Slave[SlaveIndex] = Fmy1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cy1[Step4D];
                                        Fmz1Slave[SlaveIndex] = Fmz1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cz1[Step4D];
                                        Fmx2Slave[SlaveIndex] = Fmx2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cx1[Step4D] * psxpreSlave[SlaveIndex];
                                        Fmy2Slave[SlaveIndex] = Fmy2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cy1[Step4D] * psxpreSlave[SlaveIndex];
                                        Fmz2Slave[SlaveIndex] = Fmz2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cz1[Step4D] * psxpreSlave[SlaveIndex];
                                    }
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder4D = (Step4D * nxt * nyt * nzt) + (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &spre[transferinder4D], &spreSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &psxpre[transferinder4D], &psxpreSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 2)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;

                                    simd_loadu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                    simd_loadu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                    simd_loadu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                    simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                    simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                    simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                    simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                    simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                    simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                    simd_loadu(spreSlaveV4, &spreSlave[SlaveIndex]);
                                    simd_loadu(psxpreSlaveV4, &psxpreSlave[SlaveIndex]);
                                    smx_pre1SlaveV4 = smx_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cx1V4;
                                    smy_pre1SlaveV4 = smy_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cy1V4;
                                    smz_pre1SlaveV4 = smz_pre1SlaveV4 + weight1V4 * spreSlaveV4 * cz1V4;
                                    Fmx1SlaveV4 = Fmx1SlaveV4 + weight1V4 * psxpreSlaveV4 * cx1V4;
                                    Fmy1SlaveV4 = Fmy1SlaveV4 + weight1V4 * psxpreSlaveV4 * cy1V4;
                                    Fmz1SlaveV4 = Fmz1SlaveV4 + weight1V4 * psxpreSlaveV4 * cz1V4;
                                    Fmx2SlaveV4 = Fmx2SlaveV4 + weight1V4 * psxpreSlaveV4 * cx1V4 * psxpreSlaveV4;
                                    Fmy2SlaveV4 = Fmy2SlaveV4 + weight1V4 * psxpreSlaveV4 * cy1V4 * psxpreSlaveV4;
                                    Fmz2SlaveV4 = Fmz2SlaveV4 + weight1V4 * psxpreSlaveV4 * cz1V4 * psxpreSlaveV4;

                                    simd_storeu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                    simd_storeu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                    simd_storeu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                    simd_storeu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                    simd_storeu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                    simd_storeu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                    simd_storeu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                    simd_storeu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                    simd_storeu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                }
                                for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;

                                    smx_pre1Slave[SlaveIndex] = smx_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cx1[Step4D];
                                    smy_pre1Slave[SlaveIndex] = smy_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cy1[Step4D];
                                    smz_pre1Slave[SlaveIndex] = smz_pre1Slave[SlaveIndex] + weight1[Step4D] * spreSlave[SlaveIndex] * cz1[Step4D];
                                    Fmx1Slave[SlaveIndex] = Fmx1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cx1[Step4D];
                                    Fmy1Slave[SlaveIndex] = Fmy1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cy1[Step4D];
                                    Fmz1Slave[SlaveIndex] = Fmz1Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cz1[Step4D];
                                    Fmx2Slave[SlaveIndex] = Fmx2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cx1[Step4D] * psxpreSlave[SlaveIndex];
                                    Fmy2Slave[SlaveIndex] = Fmy2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cy1[Step4D] * psxpreSlave[SlaveIndex];
                                    Fmz2Slave[SlaveIndex] = Fmz2Slave[SlaveIndex] + weight1[Step4D] * psxpreSlave[SlaveIndex] * cz1[Step4D] * psxpreSlave[SlaveIndex];
                                }
                            }
                        }
                    }
                }
                //-----------------------------------------
                // 回传
                //-----------------------------------------
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &smx_pre1Slave[getindex], &smx_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &smy_pre1Slave[getindex], &smy_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &smz_pre1Slave[getindex], &smz_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 9)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &smx_pre1Slave[getindex], &smx_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &smy_pre1Slave[getindex], &smy_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &smz_pre1Slave[getindex], &smz_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 9)
                                    ;
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &smx_pre1Slave[getindex], &smx_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &smy_pre1Slave[getindex], &smy_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &smz_pre1Slave[getindex], &smz_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                while (put_reply != 9)
                                    ;
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int getindex = conveyy * SlaveNxt;
                            put_reply = 0;
                            athread_put(PE_MODE, &smx_pre1Slave[getindex], &smx_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &smy_pre1Slave[getindex], &smy_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &smz_pre1Slave[getindex], &smz_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            while (put_reply != 9)
                                ;
                        }
                    }
                }
            }
        }
    }
}

void BoundaryF()
{
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    // 分配动态内存
    float smx_pre1Slave[slaveAll], smy_pre1Slave[slaveAll], smz_pre1Slave[slaveAll];
    memset(smx_pre1Slave, 0, slaveAll * sizeof(float));
    memset(smy_pre1Slave, 0, slaveAll * sizeof(float));
    memset(smz_pre1Slave, 0, slaveAll * sizeof(float));
    float Fmx1Slave[slaveAll], Fmy1Slave[slaveAll], Fmz1Slave[slaveAll], Fmx2Slave[slaveAll], Fmy2Slave[slaveAll], Fmz2Slave[slaveAll];
    memset(Fmx1Slave, 0, slaveAll * sizeof(float));
    memset(Fmy1Slave, 0, slaveAll * sizeof(float));
    memset(Fmz1Slave, 0, slaveAll * sizeof(float));
    memset(Fmx2Slave, 0, slaveAll * sizeof(float));
    memset(Fmy2Slave, 0, slaveAll * sizeof(float));
    memset(Fmz2Slave, 0, slaveAll * sizeof(float));
    float obstSlave[slaveAll];
    memset(obstSlave, 0, slaveAll * sizeof(float));
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                // -----------------------------------------开始计算3D--------------------------------------------------
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);

                                    athread_get(PE_MODE, &smx_pre1[transferinder], &smx_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &smy_pre1[transferinder], &smy_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &smz_pre1[transferinder], &smz_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 10)
                                        ;
                                }
                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        if (obstSlave[SlaveIndex] != 0.0f)
                                        {
                                            smx_pre1Slave[SlaveIndex] = 0.0f;
                                            smy_pre1Slave[SlaveIndex] = 0.0f;
                                            smz_pre1Slave[SlaveIndex] = 0.0f;
                                            Fmx1Slave[SlaveIndex] = 0.0f;
                                            Fmy1Slave[SlaveIndex] = 0.0f;
                                            Fmz1Slave[SlaveIndex] = 0.0f;
                                            Fmx2Slave[SlaveIndex] = 0.0f;
                                            Fmy2Slave[SlaveIndex] = 0.0f;
                                            Fmz2Slave[SlaveIndex] = 0.0f;
                                        }
                                    }
                                }

                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;
                                    put_reply = 0;
                                    athread_put(PE_MODE, &smx_pre1Slave[getindex], &smx_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &smy_pre1Slave[getindex], &smy_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &smz_pre1Slave[getindex], &smz_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 9)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);

                                athread_get(PE_MODE, &smx_pre1[transferinder], &smx_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &smy_pre1[transferinder], &smy_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &smz_pre1[transferinder], &smz_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 10)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {
                                for (ix = 0; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    if (obstSlave[SlaveIndex] != 0.0f)
                                    {
                                        smx_pre1Slave[SlaveIndex] = 0.0f;
                                        smy_pre1Slave[SlaveIndex] = 0.0f;
                                        smz_pre1Slave[SlaveIndex] = 0.0f;
                                        Fmx1Slave[SlaveIndex] = 0.0f;
                                        Fmy1Slave[SlaveIndex] = 0.0f;
                                        Fmz1Slave[SlaveIndex] = 0.0f;
                                        Fmx2Slave[SlaveIndex] = 0.0f;
                                        Fmy2Slave[SlaveIndex] = 0.0f;
                                        Fmz2Slave[SlaveIndex] = 0.0f;
                                    }
                                }
                            }

                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &smx_pre1Slave[getindex], &smx_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &smy_pre1Slave[getindex], &smy_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &smz_pre1Slave[getindex], &smz_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 9)
                                    ;
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);

                                athread_get(PE_MODE, &smx_pre1[transferinder], &smx_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &smy_pre1[transferinder], &smy_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &smz_pre1[transferinder], &smz_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 10)
                                    ;
                            }
                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < TranferPartX; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    if (obstSlave[SlaveIndex] != 0.0f)
                                    {
                                        smx_pre1Slave[SlaveIndex] = 0.0f;
                                        smy_pre1Slave[SlaveIndex] = 0.0f;
                                        smz_pre1Slave[SlaveIndex] = 0.0f;
                                        Fmx1Slave[SlaveIndex] = 0.0f;
                                        Fmy1Slave[SlaveIndex] = 0.0f;
                                        Fmz1Slave[SlaveIndex] = 0.0f;
                                        Fmx2Slave[SlaveIndex] = 0.0f;
                                        Fmy2Slave[SlaveIndex] = 0.0f;
                                        Fmz2Slave[SlaveIndex] = 0.0f;
                                    }
                                }
                            }

                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;
                                put_reply = 0;
                                athread_put(PE_MODE, &smx_pre1Slave[getindex], &smx_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &smy_pre1Slave[getindex], &smy_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &smz_pre1Slave[getindex], &smz_pre1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                while (put_reply != 9)
                                    ;
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;

                            get_reply = 0;
                            athread_get(PE_MODE, &obst[transferinder], &obstSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &smx_pre1[transferinder], &smx_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &smy_pre1[transferinder], &smy_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &smz_pre1[transferinder], &smz_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 10)
                                ;
                        }
                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {
                            for (ix = 0; ix < SlaveNxt; ix++)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                if (obstSlave[SlaveIndex] != 0.0f)
                                {
                                    smx_pre1Slave[SlaveIndex] = 0.0f;
                                    smy_pre1Slave[SlaveIndex] = 0.0f;
                                    smz_pre1Slave[SlaveIndex] = 0.0f;
                                    Fmx1Slave[SlaveIndex] = 0.0f;
                                    Fmy1Slave[SlaveIndex] = 0.0f;
                                    Fmz1Slave[SlaveIndex] = 0.0f;
                                    Fmx2Slave[SlaveIndex] = 0.0f;
                                    Fmy2Slave[SlaveIndex] = 0.0f;
                                    Fmz2Slave[SlaveIndex] = 0.0f;
                                }
                            }
                        }

                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int getindex = conveyy * SlaveNxt;
                            put_reply = 0;
                            athread_put(PE_MODE, &smx_pre1Slave[getindex], &smx_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &smy_pre1Slave[getindex], &smy_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &smz_pre1Slave[getindex], &smz_pre1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            while (put_reply != 9)
                                ;
                        }
                    }
                }
            }
        }
    }
}

void UpdateF()
{
    float a1 = -1.0f;
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 20;
    SlaveNyt = 20;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    // 分配动态内存
    float smx_pre1Slave[slaveAll], smy_pre1Slave[slaveAll], smz_pre1Slave[slaveAll];
    memset(smx_pre1Slave, 0, slaveAll * sizeof(float));
    memset(smy_pre1Slave, 0, slaveAll * sizeof(float));
    memset(smz_pre1Slave, 0, slaveAll * sizeof(float));
    float Fmx1Slave[slaveAll], Fmy1Slave[slaveAll], Fmz1Slave[slaveAll], Fmx2Slave[slaveAll], Fmy2Slave[slaveAll], Fmz2Slave[slaveAll];
    memset(Fmx1Slave, 0, slaveAll * sizeof(float));
    memset(Fmy1Slave, 0, slaveAll * sizeof(float));
    memset(Fmz1Slave, 0, slaveAll * sizeof(float));
    memset(Fmx2Slave, 0, slaveAll * sizeof(float));
    memset(Fmy2Slave, 0, slaveAll * sizeof(float));
    memset(Fmz2Slave, 0, slaveAll * sizeof(float));
    float F1sxSlave[slaveAll], F1sySlave[slaveAll], F1szSlave[slaveAll];
    memset(F1sxSlave, 0, slaveAll * sizeof(float));
    memset(F1sySlave, 0, slaveAll * sizeof(float));
    memset(F1szSlave, 0, slaveAll * sizeof(float));
    float psxSlave[slaveAll];
    memset(psxSlave, 0, slaveAll * sizeof(float));
    floatv4 psxSlaveV4;
    floatv4 smx_pre1SlaveV4, smy_pre1SlaveV4, smz_pre1SlaveV4;
    floatv4 Fmx1SlaveV4, Fmy1SlaveV4, Fmz1SlaveV4, Fmx2SlaveV4, Fmy2SlaveV4, Fmz2SlaveV4;
    floatv4 F1sxSlaveV4, F1sySlaveV4, F1szSlaveV4;
    // 开始计算
    for (StepSlaveZ = 0; StepSlaveZ < nzt; StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                int TranferPartX = (_COL + 1) * SlaveNxt - (nxt - StepSlaveX);
                int TranferParty = (_ROW + 1) * SlaveNyt - (nyt - StepSlaveY);
                if ((_ROW + 1) * SlaveNyt > (nyt - StepSlaveY)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNyt个而是4个
                {
                    if ((_ROW + 1) * SlaveNyt - (nyt - StepSlaveY) < SlaveNyt) // 4 这种非整除的 传到conveyy=(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                    {
                        if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                        {
                            if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                            {
                                //*************************************************************************************************
                                // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                                //*************************************************************************************************
                                // -----------------------------------------开始计算3D--------------------------------------------------
                                //-----------------------------------------
                                // 取数组
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                    // 本层取值
                                    // 传输数据DMA对应在模型上的坐标索引
                                    // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                    // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                    // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                    // (StepSlaveX) 到索引所在的核组的首地址
                                    // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                    // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                    // (conveyy * nxt) 到索引所在的列了
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                    // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                    int getindex = conveyy * SlaveNxt;

                                    get_reply = 0;
                                    athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &smx_pre1[transferinder], &smx_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &smy_pre1[transferinder], &smy_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &smz_pre1[transferinder], &smz_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                    while (get_reply != 10)
                                        ;
                                }

                                //-----------------------------------------
                                // 计算
                                //-----------------------------------------
                                for (iy = 0; iy < TranferParty; iy++)
                                {
                                    for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        simd_loadu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                        simd_loadu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                        simd_loadu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                        simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                        simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                        simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                        simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                        simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                        simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                        simd_loadu(psxSlaveV4, &psxSlave[SlaveIndex]);
                                        simd_loadu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                        simd_loadu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                        simd_loadu(F1szSlaveV4, &F1szSlave[SlaveIndex]);

                                        F1sxSlaveV4 = -psxSlaveV4 * psxSlaveV4 * smx_pre1SlaveV4 * G2; // 流固作用力
                                        F1sySlaveV4 = -psxSlaveV4 * psxSlaveV4 * smy_pre1SlaveV4 * G2;
                                        F1szSlaveV4 = -psxSlaveV4 * psxSlaveV4 * smz_pre1SlaveV4 * G2;

                                        Fmx1SlaveV4 = -psxSlaveV4 * Fmx1SlaveV4 * G1 / 3.0f * B; // 流体内部作用力
                                        Fmy1SlaveV4 = -psxSlaveV4 * Fmy1SlaveV4 * G1 / 3.0f * B;
                                        Fmz1SlaveV4 = -psxSlaveV4 * Fmz1SlaveV4 * G1 / 3.0f * B;
                                        Fmx2SlaveV4 = -Fmx2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;
                                        Fmy2SlaveV4 = -Fmy2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;
                                        Fmz2SlaveV4 = -Fmz2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;

                                        simd_storeu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                        simd_storeu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                        simd_storeu(F1szSlaveV4, &F1szSlave[SlaveIndex]);
                                        simd_storeu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                        simd_storeu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                        simd_storeu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                        simd_storeu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                        simd_storeu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                        simd_storeu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                    }
                                    for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                    {
                                        int SlaveIndex = iy * SlaveNxt + ix;
                                        F1sxSlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smx_pre1Slave[SlaveIndex] * G2; // 流固作用力
                                        F1sySlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smy_pre1Slave[SlaveIndex] * G2;
                                        F1szSlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smz_pre1Slave[SlaveIndex] * G2;

                                        Fmx1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmx1Slave[SlaveIndex] * G1 / 3.0f * B; // 流体内部作用力
                                        Fmy1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmy1Slave[SlaveIndex] * G1 / 3.0f * B;
                                        Fmz1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmz1Slave[SlaveIndex] * G1 / 3.0f * B;
                                        Fmx2Slave[SlaveIndex] = -Fmx2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                                        Fmy2Slave[SlaveIndex] = -Fmy2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                                        Fmz2Slave[SlaveIndex] = -Fmz2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                                    }
                                }
                                //-----------------------------------------
                                // 回传
                                //-----------------------------------------
                                for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                                {
                                    int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                    int getindex = conveyy * SlaveNxt;

                                    put_reply = 0;
                                    athread_put(PE_MODE, &F1sxSlave[getindex], &F1sx[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &F1sySlave[getindex], &F1sy[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &F1szSlave[getindex], &F1sz[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                    while (put_reply != 9)
                                        ;
                                }
                            }
                        }
                        if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                        {
                            //*************************************************************************************************
                            // x 传SlaveNxt   y 传到(_ROW + 1) * SlaveNyt - (nyt - StepSlaveY)
                            //*************************************************************************************************
                            // -----------------------------------------开始计算3D--------------------------------------------------
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &smx_pre1[transferinder], &smx_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &smy_pre1[transferinder], &smy_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &smz_pre1[transferinder], &smz_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                                while (get_reply != 10)
                                    ;
                            }

                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < TranferParty; iy++)
                            {
                                for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    simd_loadu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                    simd_loadu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                    simd_loadu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                    simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                    simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                    simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                    simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                    simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                    simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                    simd_loadu(psxSlaveV4, &psxSlave[SlaveIndex]);
                                    simd_loadu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                    simd_loadu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                    simd_loadu(F1szSlaveV4, &F1szSlave[SlaveIndex]);

                                    F1sxSlaveV4 = -psxSlaveV4 * psxSlaveV4 * smx_pre1SlaveV4 * G2; // 流固作用力
                                    F1sySlaveV4 = -psxSlaveV4 * psxSlaveV4 * smy_pre1SlaveV4 * G2;
                                    F1szSlaveV4 = -psxSlaveV4 * psxSlaveV4 * smz_pre1SlaveV4 * G2;

                                    Fmx1SlaveV4 = -psxSlaveV4 * Fmx1SlaveV4 * G1 / 3.0f * B; // 流体内部作用力
                                    Fmy1SlaveV4 = -psxSlaveV4 * Fmy1SlaveV4 * G1 / 3.0f * B;
                                    Fmz1SlaveV4 = -psxSlaveV4 * Fmz1SlaveV4 * G1 / 3.0f * B;
                                    Fmx2SlaveV4 = -Fmx2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;
                                    Fmy2SlaveV4 = -Fmy2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;
                                    Fmz2SlaveV4 = -Fmz2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;

                                    simd_storeu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                    simd_storeu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                    simd_storeu(F1szSlaveV4, &F1szSlave[SlaveIndex]);
                                    simd_storeu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                    simd_storeu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                    simd_storeu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                    simd_storeu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                    simd_storeu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                    simd_storeu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                }
                                for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    F1sxSlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smx_pre1Slave[SlaveIndex] * G2; // 流固作用力
                                    F1sySlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smy_pre1Slave[SlaveIndex] * G2;
                                    F1szSlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smz_pre1Slave[SlaveIndex] * G2;

                                    Fmx1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmx1Slave[SlaveIndex] * G1 / 3.0f * B; // 流体内部作用力
                                    Fmy1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmy1Slave[SlaveIndex] * G1 / 3.0f * B;
                                    Fmz1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmz1Slave[SlaveIndex] * G1 / 3.0f * B;
                                    Fmx2Slave[SlaveIndex] = -Fmx2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                                    Fmy2Slave[SlaveIndex] = -Fmy2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                                    Fmz2Slave[SlaveIndex] = -Fmz2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < TranferParty; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;

                                put_reply = 0;
                                athread_put(PE_MODE, &F1sxSlave[getindex], &F1sx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &F1sySlave[getindex], &F1sy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &F1szSlave[getindex], &F1sz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 9)
                                    ;
                            }
                        }
                    }
                }
                if ((_ROW + 1) * SlaveNyt <= (nyt - StepSlaveY)) // 可以传到conveyy=slaveNyt
                {
                    if ((_COL + 1) * SlaveNxt > (nxt - StepSlaveX)) // 比如64这种数据 那么需要第7个核进行计算 但传递数据不能传SlaveNxt个而是4个
                    {
                        if ((_COL + 1) * SlaveNxt - (nxt - StepSlaveX) < SlaveNxt) // 4这种非整除的 传递的个数是(_COL + 1) * SlaveNxt - (nxt - StepSlavex)
                        {
                            //*************************************************************************************************
                            // x 传(_COL + 1) * SlaveNxt - (nxt - StepSlavex)   y 传到SlaveNyt
                            //*************************************************************************************************
                            // -----------------------------------------开始计算3D--------------------------------------------------
                            //-----------------------------------------
                            // 取数组
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                                // 本层取值
                                // 传输数据DMA对应在模型上的坐标索引
                                // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                                // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                                // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt;

                                get_reply = 0;
                                athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &smx_pre1[transferinder], &smx_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &smy_pre1[transferinder], &smy_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &smz_pre1[transferinder], &smz_pre1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * TranferPartX, &get_reply, 0, 0, 0);
                                while (get_reply != 10)
                                    ;
                            }

                            //-----------------------------------------
                            // 计算
                            //-----------------------------------------
                            for (iy = 0; iy < SlaveNyt; iy++)
                            {
                                for (ix = 0; ix < (TranferPartX / 4) * 4; ix += 4)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    simd_loadu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                    simd_loadu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                    simd_loadu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                    simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                    simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                    simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                    simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                    simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                    simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                    simd_loadu(psxSlaveV4, &psxSlave[SlaveIndex]);
                                    simd_loadu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                    simd_loadu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                    simd_loadu(F1szSlaveV4, &F1szSlave[SlaveIndex]);

                                    F1sxSlaveV4 = -psxSlaveV4 * psxSlaveV4 * smx_pre1SlaveV4 * G2; // 流固作用力
                                    F1sySlaveV4 = -psxSlaveV4 * psxSlaveV4 * smy_pre1SlaveV4 * G2;
                                    F1szSlaveV4 = -psxSlaveV4 * psxSlaveV4 * smz_pre1SlaveV4 * G2;

                                    Fmx1SlaveV4 = -psxSlaveV4 * Fmx1SlaveV4 * G1 / 3.0f * B; // 流体内部作用力
                                    Fmy1SlaveV4 = -psxSlaveV4 * Fmy1SlaveV4 * G1 / 3.0f * B;
                                    Fmz1SlaveV4 = -psxSlaveV4 * Fmz1SlaveV4 * G1 / 3.0f * B;
                                    Fmx2SlaveV4 = -Fmx2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;
                                    Fmy2SlaveV4 = -Fmy2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;
                                    Fmz2SlaveV4 = -Fmz2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;

                                    simd_storeu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                    simd_storeu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                    simd_storeu(F1szSlaveV4, &F1szSlave[SlaveIndex]);
                                    simd_storeu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                    simd_storeu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                    simd_storeu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                    simd_storeu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                    simd_storeu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                    simd_storeu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                }
                                for (ix = (TranferPartX / 4) * 4; ix < TranferPartX; ix++)
                                {
                                    int SlaveIndex = iy * SlaveNxt + ix;
                                    F1sxSlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smx_pre1Slave[SlaveIndex] * G2; // 流固作用力
                                    F1sySlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smy_pre1Slave[SlaveIndex] * G2;
                                    F1szSlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smz_pre1Slave[SlaveIndex] * G2;

                                    Fmx1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmx1Slave[SlaveIndex] * G1 / 3.0f * B; // 流体内部作用力
                                    Fmy1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmy1Slave[SlaveIndex] * G1 / 3.0f * B;
                                    Fmz1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmz1Slave[SlaveIndex] * G1 / 3.0f * B;
                                    Fmx2Slave[SlaveIndex] = -Fmx2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                                    Fmy2Slave[SlaveIndex] = -Fmy2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                                    Fmz2Slave[SlaveIndex] = -Fmz2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                                }
                            }
                            //-----------------------------------------
                            // 回传
                            //-----------------------------------------
                            for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                            {
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                int getindex = conveyy * SlaveNxt;

                                put_reply = 0;
                                athread_put(PE_MODE, &F1sxSlave[getindex], &F1sx[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &F1sySlave[getindex], &F1sy[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &F1szSlave[getindex], &F1sz[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * TranferPartX, &put_reply, 0, 0);
                                while (put_reply != 9)
                                    ;
                            }
                        }
                    }
                    if ((_COL + 1) * SlaveNxt <= (nxt - StepSlaveX)) // 数据足够多核内每次传SlaveNxt个数
                    {
                        //*************************************************************************************************
                        // x 传SlaveNxt   y 传到SlaveNyt
                        //*************************************************************************************************
                        // -----------------------------------------开始计算3D--------------------------------------------------
                        //-----------------------------------------
                        // 取数组
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            //------------------------------------------------传输数据DMA对应在模型上的坐标索引--------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (Step4D * nxt * nyt * nzt) 对应第四维度上的首地址
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);

                            // 每个从核中所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex = conveyy * SlaveNxt;

                            get_reply = 0;
                            athread_get(PE_MODE, &psx[transferinder], &psxSlave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &smx_pre1[transferinder], &smx_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &smy_pre1[transferinder], &smy_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &smz_pre1[transferinder], &smz_pre1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmx1[transferinder], &Fmx1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmy1[transferinder], &Fmy1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmz1[transferinder], &Fmz1Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmx2[transferinder], &Fmx2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmy2[transferinder], &Fmy2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &Fmz2[transferinder], &Fmz2Slave[getindex], sizeof(float) * SlaveNxt, &get_reply, 0, 0, 0);
                            while (get_reply != 10)
                                ;
                        }

                        //-----------------------------------------
                        // 计算
                        //-----------------------------------------
                        for (iy = 0; iy < SlaveNyt; iy++)
                        {
                            for (ix = 0; ix < (SlaveNxt / 4) * 4; ix += 4)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                simd_loadu(smx_pre1SlaveV4, &smx_pre1Slave[SlaveIndex]);
                                simd_loadu(smy_pre1SlaveV4, &smy_pre1Slave[SlaveIndex]);
                                simd_loadu(smz_pre1SlaveV4, &smz_pre1Slave[SlaveIndex]);
                                simd_loadu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                simd_loadu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                simd_loadu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                simd_loadu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                simd_loadu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                simd_loadu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                                simd_loadu(psxSlaveV4, &psxSlave[SlaveIndex]);
                                simd_loadu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                simd_loadu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                simd_loadu(F1szSlaveV4, &F1szSlave[SlaveIndex]);

                                F1sxSlaveV4 = -psxSlaveV4 * psxSlaveV4 * smx_pre1SlaveV4 * G2; // 流固作用力
                                F1sySlaveV4 = -psxSlaveV4 * psxSlaveV4 * smy_pre1SlaveV4 * G2;
                                F1szSlaveV4 = -psxSlaveV4 * psxSlaveV4 * smz_pre1SlaveV4 * G2;

                                Fmx1SlaveV4 = -psxSlaveV4 * Fmx1SlaveV4 * G1 / 3.0f * B; // 流体内部作用力
                                Fmy1SlaveV4 = -psxSlaveV4 * Fmy1SlaveV4 * G1 / 3.0f * B;
                                Fmz1SlaveV4 = -psxSlaveV4 * Fmz1SlaveV4 * G1 / 3.0f * B;
                                Fmx2SlaveV4 = -Fmx2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;
                                Fmy2SlaveV4 = -Fmy2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;
                                Fmz2SlaveV4 = -Fmz2SlaveV4 * G1 / 3.0f * (1.0f - B) / 2.0f;

                                simd_storeu(F1sxSlaveV4, &F1sxSlave[SlaveIndex]);
                                simd_storeu(F1sySlaveV4, &F1sySlave[SlaveIndex]);
                                simd_storeu(F1szSlaveV4, &F1szSlave[SlaveIndex]);
                                simd_storeu(Fmx1SlaveV4, &Fmx1Slave[SlaveIndex]);
                                simd_storeu(Fmy1SlaveV4, &Fmy1Slave[SlaveIndex]);
                                simd_storeu(Fmz1SlaveV4, &Fmz1Slave[SlaveIndex]);
                                simd_storeu(Fmx2SlaveV4, &Fmx2Slave[SlaveIndex]);
                                simd_storeu(Fmy2SlaveV4, &Fmy2Slave[SlaveIndex]);
                                simd_storeu(Fmz2SlaveV4, &Fmz2Slave[SlaveIndex]);
                            }
                            for (ix = (SlaveNxt / 4) * 4; ix < SlaveNxt; ix++)
                            {
                                int SlaveIndex = iy * SlaveNxt + ix;
                                F1sxSlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smx_pre1Slave[SlaveIndex] * G2; // 流固作用力
                                F1sySlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smy_pre1Slave[SlaveIndex] * G2;
                                F1szSlave[SlaveIndex] = -psxSlave[SlaveIndex] * psxSlave[SlaveIndex] * smz_pre1Slave[SlaveIndex] * G2;

                                Fmx1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmx1Slave[SlaveIndex] * G1 / 3.0f * B; // 流体内部作用力
                                Fmy1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmy1Slave[SlaveIndex] * G1 / 3.0f * B;
                                Fmz1Slave[SlaveIndex] = -psxSlave[SlaveIndex] * Fmz1Slave[SlaveIndex] * G1 / 3.0f * B;
                                Fmx2Slave[SlaveIndex] = -Fmx2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                                Fmy2Slave[SlaveIndex] = -Fmy2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                                Fmz2Slave[SlaveIndex] = -Fmz2Slave[SlaveIndex] * G1 / 3.0f * (1.0f - B) / 2.0f;
                            }
                        }
                        //-----------------------------------------
                        // 回传
                        //-----------------------------------------
                        for (int conveyy = 0; conveyy < SlaveNyt; conveyy++)
                        {
                            int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            int getindex = conveyy * SlaveNxt;

                            put_reply = 0;
                            athread_put(PE_MODE, &F1sxSlave[getindex], &F1sx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &F1sySlave[getindex], &F1sy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &F1szSlave[getindex], &F1sz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmx1Slave[getindex], &Fmx1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmy1Slave[getindex], &Fmy1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmz1Slave[getindex], &Fmz1[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmx2Slave[getindex], &Fmx2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmy2Slave[getindex], &Fmy2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            athread_put(PE_MODE, &Fmz2Slave[getindex], &Fmz2[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                            while (put_reply != 9)
                                ;
                        }
                    }
                }
            }
        }
    }
}
