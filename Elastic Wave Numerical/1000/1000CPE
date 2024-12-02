#include <stdio.h>
#include <math.h>
#include <string.h>
#include <slave.h>
#include <simd.h>
// #define LWPF_UNIT U(test)
// #define LWPF_KERNELS K(A)
// #define EVT_PC0 PC0_CYCLE
// #define EVT_PC1 PC1_INST_LDMACCESS
// #define EVT_PC2 PC2_L1IC_MISSTIME // PC2_GLD
// // #define EVT_PC2 PC2_GLD
// #define EVT_PC3 PC3_DCACHE_HIT_LDM
// #define EVT_PC4 PC4_DCACHE_ACCESS
// #define EVT_PC5 PC5_DCACHE_MISS
// #include "lwpf3/lwpf.h"

// 每个从核各方向的大小
__thread_local float xFHalfTemp2, xFIntTemp2, yFHalfTemp2, yFIntTemp2, zFHalfTemp2, zFIntTemp2;
__thread_local int ix, iy, iz;
extern float *rho_ext, *txx, *tyy, *tzz, *txy, *txz, *tyz, *dxi, *dxi2, *dyj, *dyj2, *dzk, *dzk2, *vx, *vy, *vz, *mu_ext, *lamda2u_ext, *lamda_ext, *muxy, *muxz, *muyz, *edxi, *edxi2, *edyj, *edyj2, *edzk, *edzk2;
extern float H, DT;
extern int nxt, nyt, nzt;
extern int myid, it;
// extern float *dxii, *vxaccept, *txxaccept, *rhoaccept, *mu_extaccept;
extern float *pmlxSxx, *pmlySxy, *pmlzSxz, *pmlxSxy, *pmlySyy, *pmlzSyz, *pmlxSxz, *pmlySyz, *pmlzSzz;
extern float *DxSxxPre, *DySxyPre, *DzSxzPre, *DxSxyPre, *DySyyPre, *DzSyzPre, *DxSxzPre, *DySyzPre, *DzSzzPre;
extern float *pmlxVx, *pmlyVy, *pmlzVz, *pmlzVx, *pmlxVz, *pmlyVx, *pmlxVy, *pmlzVy, *pmlyVz;
extern float *DxVxPre, *DyVyPre, *DzVzPre, *DyVxPre, *DxVyPre, *DzVxPre, *DxVzPre, *DzVyPre, *DyVzPre;
volatile __thread_local int get_reply, put_reply;
//**********************************************************************************************************************************************************************
//                                                          计算速度
//**********************************************************************************************************************************************************************

void funVelocity()
{
    // lwpf_enter(test);vim
    // lwpf_start(A);
    int StepSlaveX, StepSlaveY, StepSlaveZ;
    // define the middle variable in slave
    float rho_tempx, rho_tempy, rho_tempz, xx1, xy1, xz1, yx2, yy2, yz2, zx3, zy3, zz3;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 10;
    SlaveNyt = 10;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // lwpf_stop(A);

    // 单个从核中的计算长度多传的外壳
    int SlaveNxt1, SlaveNyt1, SlaveNzt1, slaveAll1, slaveAll2, slaveAll3;
    SlaveNxt1 = SlaveNxt + 2;
    SlaveNyt1 = SlaveNyt + 2;
    SlaveNzt1 = SlaveNzt + 2;
    slaveAll3 = SlaveNxt1 * SlaveNyt1 * SlaveNzt1;
    slaveAll2 = SlaveNxt1 * SlaveNyt1 * 2;
    slaveAll1 = SlaveNxt1 * SlaveNyt1;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    // 在从核中开辟空间
    // 需要1层的
    float pmlxSxxSlave[slaveAll1], pmlySxySlave[slaveAll1], pmlzSxzSlave[slaveAll1], pmlxSxySlave[slaveAll1], pmlySyySlave[slaveAll1], pmlzSyzSlave[slaveAll1], pmlxSxzSlave[slaveAll1], pmlySyzSlave[slaveAll1], pmlzSzzSlave[slaveAll1];
    float DxSxxPreSlave[slaveAll1], DySxyPreSlave[slaveAll1], DzSxzPreSlave[slaveAll1], DxSxyPreSlave[slaveAll1], DySyyPreSlave[slaveAll1], DzSyzPreSlave[slaveAll1], DxSxzPreSlave[slaveAll1], DySyzPreSlave[slaveAll1], DzSzzPreSlave[slaveAll1];
    float vxSlave[slaveAll1], vySlave[slaveAll1], vzSlave[slaveAll1];
    float dxiSlave[slaveAll1], dxi2Slave[slaveAll1], dyjSlave[slaveAll1], dyj2Slave[slaveAll1], dzkSlave[slaveAll1], dzk2Slave[slaveAll1];
    float vxSlaveresult[slaveAll1], vySlaveresult[slaveAll1], vzSlaveresult[slaveAll1];
    float txxSlave[slaveAll1], tyySlave[slaveAll1], txySlave[slaveAll1];
    float edxiSlave[slaveAll1], edxi2Slave[slaveAll1], edyjSlave[slaveAll1], edyj2Slave[slaveAll1], edzkSlave[slaveAll1], edzk2Slave[slaveAll1];
    // 需要两层的
    float tzzSlaveUp[slaveAll1], txzSlaveUp[slaveAll1], tyzSlaveUp[slaveAll1];
    float rho_extSlaveUp[slaveAll1];
    float tzzSlaveDown[slaveAll1], txzSlaveDown[slaveAll1], tyzSlaveDown[slaveAll1];
    float rho_extSlaveDown[slaveAll1];
    // 进行掩盖赋值时需要的中间变量
    float *UPtzzSlave, *DOWNtzzSlave, *MIDDLEtzzSlave;
    float *UPtxzSlave, *DOWNtxzSlave, *MIDDLEtxzSlave;
    float *UPtyzSlave, *DOWNtyzSlave, *MIDDLEtyzSlave;
    float *UPrho_extSlave, *DOWNrho_extSlave, *MIDDLErho_extSlave;
    // 一层
    memset(pmlxSxxSlave, 0, slaveAll1 * sizeof(float));
    memset(pmlySxySlave, 0, slaveAll1 * sizeof(float));
    memset(pmlzSxzSlave, 0, slaveAll1 * sizeof(float));
    memset(pmlxSxySlave, 0, slaveAll1 * sizeof(float));
    memset(pmlySyySlave, 0, slaveAll1 * sizeof(float));
    memset(pmlzSyzSlave, 0, slaveAll1 * sizeof(float));
    memset(pmlxSxzSlave, 0, slaveAll1 * sizeof(float));
    memset(pmlySyzSlave, 0, slaveAll1 * sizeof(float));
    memset(pmlzSzzSlave, 0, slaveAll1 * sizeof(float));
    memset(DxSxxPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DySxyPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DzSxzPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DxSxyPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DySyyPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DzSyzPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DxSxzPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DySyzPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DzSzzPreSlave, 0, slaveAll1 * sizeof(float));
    memset(vxSlave, 0, slaveAll1 * sizeof(float));
    memset(vySlave, 0, slaveAll1 * sizeof(float));
    memset(vzSlave, 0, slaveAll1 * sizeof(float));
    memset(dxiSlave, 0, slaveAll1 * sizeof(float));
    memset(dxi2Slave, 0, slaveAll1 * sizeof(float));
    memset(dyjSlave, 0, slaveAll1 * sizeof(float));
    memset(dyj2Slave, 0, slaveAll1 * sizeof(float));
    memset(dzkSlave, 0, slaveAll1 * sizeof(float));
    memset(dzk2Slave, 0, slaveAll1 * sizeof(float));
    memset(vxSlaveresult, 0, slaveAll1 * sizeof(float));
    memset(vySlaveresult, 0, slaveAll1 * sizeof(float));
    memset(vzSlaveresult, 0, slaveAll1 * sizeof(float));
    memset(txxSlave, 0, slaveAll1 * sizeof(float));
    memset(tyySlave, 0, slaveAll1 * sizeof(float));
    memset(txySlave, 0, slaveAll1 * sizeof(float));
    memset(edxiSlave, 0, slaveAll1 * sizeof(float));
    memset(edxi2Slave, 0, slaveAll1 * sizeof(float));
    memset(edyjSlave, 0, slaveAll1 * sizeof(float));
    memset(edyj2Slave, 0, slaveAll1 * sizeof(float));
    memset(edzkSlave, 0, slaveAll1 * sizeof(float));
    memset(edzk2Slave, 0, slaveAll1 * sizeof(float));
    // 两层
    memset(rho_extSlaveUp, 0, slaveAll1 * sizeof(float));
    memset(tzzSlaveUp, 0, slaveAll1 * sizeof(float));
    memset(txzSlaveUp, 0, slaveAll1 * sizeof(float));
    memset(tyzSlaveUp, 0, slaveAll1 * sizeof(float));
    memset(rho_extSlaveDown, 0, slaveAll1 * sizeof(float));
    memset(tzzSlaveDown, 0, slaveAll1 * sizeof(float));
    memset(txzSlaveDown, 0, slaveAll1 * sizeof(float));
    memset(tyzSlaveDown, 0, slaveAll1 * sizeof(float));
    // 进行赋值掩盖
    UPtzzSlave = tzzSlaveUp;
    DOWNtzzSlave = tzzSlaveDown;
    UPtxzSlave = txzSlaveUp;
    DOWNtxzSlave = txzSlaveDown;
    UPtyzSlave = tyzSlaveUp;
    DOWNtyzSlave = tyzSlaveDown;
    UPrho_extSlave = rho_extSlaveUp;
    DOWNrho_extSlave = rho_extSlaveDown;
    int limitnxt, limitnyt;
    limitnxt = ceil((nxt / (SlaveNxt * slaveXnumb)) - 1) * (SlaveNxt * slaveXnumb);
    limitnyt = ceil((nyt / (SlaveNyt * slaveYnumb)) - 1) * (SlaveNyt * slaveYnumb);

    // 进行从核的跳步  从核分配为8*8*1 当64个从核内走完一次循环后，进行整体的跳步，对模型的下一块进行计算
    for (StepSlaveZ = 0; StepSlaveZ < (nzt - 1); StepSlaveZ++)
    {
        for (StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                //------------------------------------------读数据--------------------------------------------------------------
                // 从主核中取数组DMA传输   每次从主核多读一行，在主核中表现为每个从核间取值有交叉
                //*******************************************************************************************************************
                // 一层数值  iz=0不用取
                for (int conveyy = 0; conveyy < SlaveNyt1; conveyy++)
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
                    int transferinder1 = (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                    // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                    int getindex1 = conveyy * SlaveNxt1;

                    if ((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3)))
                    {
                    }
                    if (!((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3))))
                    {
                        if (_COL == 7 && (StepSlaveX == limitnxt)) // 对于模型的nxt位置 只取slavenxt1-1个值
                        {
                            // //9A
                            // athread_dma_get(&vxSlave[getindex1], &vx[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&vySlave[getindex1], &vy[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&vzSlave[getindex1], &vz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dxiSlave[getindex1], &dxi[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dxi2Slave[getindex1], &dxi2[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dyjSlave[getindex1], &dyj[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dyj2Slave[getindex1], &dyj2[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dzkSlave[getindex1], &dzk[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dzk2Slave[getindex1], &dzk2[transferinder1], sizeof(float) * (SlaveNxt1 - 2));

                            // athread_dma_get(&pmlxSxxSlave[getindex1], &pmlxSxx[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlySxySlave[getindex1], &pmlySxy[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlzSxzSlave[getindex1], &pmlzSxz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlxSxySlave[getindex1], &pmlxSxy[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlySyySlave[getindex1], &pmlySyy[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlzSyzSlave[getindex1], &pmlzSyz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlxSxzSlave[getindex1], &pmlxSxz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlySyzSlave[getindex1], &pmlySyz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlzSzzSlave[getindex1], &pmlzSzz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));

                            // athread_dma_get(&DxSxxPreSlave[getindex1], &DxSxxPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DySxyPreSlave[getindex1], &DySxyPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DzSxzPreSlave[getindex1], &DzSxzPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DxSxyPreSlave[getindex1], &DxSxyPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DySyyPreSlave[getindex1], &DySyyPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DzSyzPreSlave[getindex1], &DzSyzPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DxSxzPreSlave[getindex1], &DxSxzPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DySyzPreSlave[getindex1], &DySyzPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DzSzzPreSlave[getindex1], &DzSzzPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));

                            // athread_dma_get(&txxSlave[getindex1], &txx[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&tyySlave[getindex1], &tyy[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&txySlave[getindex1], &txy[transferinder1], sizeof(float) * (SlaveNxt1 - 2));

                            // 5A
                            get_reply = 0;
                            athread_get(PE_MODE, &vx[transferinder1], &vxSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &vy[transferinder1], &vySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &vz[transferinder1], &vzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dxi[transferinder1], &dxiSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dxi2[transferinder1], &dxi2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dyj[transferinder1], &dyjSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dyj2[transferinder1], &dyj2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dzk[transferinder1], &dzkSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dzk2[transferinder1], &dzk2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &pmlxSxx[transferinder1], &pmlxSxxSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlySxy[transferinder1], &pmlySxySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzSxz[transferinder1], &pmlzSxzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlxSxy[transferinder1], &pmlxSxySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlySyy[transferinder1], &pmlySyySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzSyz[transferinder1], &pmlzSyzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlxSxz[transferinder1], &pmlxSxzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlySyz[transferinder1], &pmlySyzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzSzz[transferinder1], &pmlzSzzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &DxSxxPre[transferinder1], &DxSxxPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DySxyPre[transferinder1], &DySxyPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzSxzPre[transferinder1], &DzSxzPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DxSxyPre[transferinder1], &DxSxyPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DySyyPre[transferinder1], &DySyyPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzSyzPre[transferinder1], &DzSyzPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DxSxzPre[transferinder1], &DxSxzPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DySyzPre[transferinder1], &DySyzPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzSzzPre[transferinder1], &DzSzzPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &txx[transferinder1], &txxSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &tyy[transferinder1], &tyySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &txy[transferinder1], &txySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &edxi[transferinder1], &edxiSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edxi2[transferinder1], &edxi2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edyj[transferinder1], &edyjSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edyj2[transferinder1], &edyj2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edzk[transferinder1], &edzkSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edzk2[transferinder1], &edzk2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            while (get_reply != 36)
                                ;
                        }
                        if (!(_COL == 7 && (StepSlaveX == limitnxt)))
                        {
                            // 9A
                            // athread_dma_get(&vxSlave[getindex1], &vx[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&vySlave[getindex1], &vy[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&vzSlave[getindex1], &vz[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dxiSlave[getindex1], &dxi[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dxi2Slave[getindex1], &dxi2[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dyjSlave[getindex1], &dyj[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dyj2Slave[getindex1], &dyj2[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dzkSlave[getindex1], &dzk[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dzk2Slave[getindex1], &dzk2[transferinder1], sizeof(float) * SlaveNxt1);

                            // athread_dma_get(&pmlxSxxSlave[getindex1], &pmlxSxx[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlySxySlave[getindex1], &pmlySxy[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlzSxzSlave[getindex1], &pmlzSxz[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlxSxySlave[getindex1], &pmlxSxy[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlySyySlave[getindex1], &pmlySyy[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlzSyzSlave[getindex1], &pmlzSyz[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlxSxzSlave[getindex1], &pmlxSxz[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlySyzSlave[getindex1], &pmlySyz[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlzSzzSlave[getindex1], &pmlzSzz[transferinder1], sizeof(float) * SlaveNxt1);

                            // athread_dma_get(&DxSxxPreSlave[getindex1], &DxSxxPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DySxyPreSlave[getindex1], &DySxyPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DzSxzPreSlave[getindex1], &DzSxzPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DxSxyPreSlave[getindex1], &DxSxyPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DySyyPreSlave[getindex1], &DySyyPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DzSyzPreSlave[getindex1], &DzSyzPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DxSxzPreSlave[getindex1], &DxSxzPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DySyzPreSlave[getindex1], &DySyzPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DzSzzPreSlave[getindex1], &DzSzzPre[transferinder1], sizeof(float) * SlaveNxt1);

                            // athread_dma_get(&txxSlave[getindex1], &txx[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&tyySlave[getindex1], &tyy[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&txySlave[getindex1], &txy[transferinder1], sizeof(float) * SlaveNxt1);
                            // 5A
                            get_reply = 0;
                            athread_get(PE_MODE, &vx[transferinder1], &vxSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &vy[transferinder1], &vySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &vz[transferinder1], &vzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dxi[transferinder1], &dxiSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dxi2[transferinder1], &dxi2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dyj[transferinder1], &dyjSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dyj2[transferinder1], &dyj2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dzk[transferinder1], &dzkSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dzk2[transferinder1], &dzk2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &pmlxSxx[transferinder1], &pmlxSxxSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlySxy[transferinder1], &pmlySxySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzSxz[transferinder1], &pmlzSxzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlxSxy[transferinder1], &pmlxSxySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlySyy[transferinder1], &pmlySyySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzSyz[transferinder1], &pmlzSyzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlxSxz[transferinder1], &pmlxSxzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlySyz[transferinder1], &pmlySyzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzSzz[transferinder1], &pmlzSzzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &DxSxxPre[transferinder1], &DxSxxPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DySxyPre[transferinder1], &DySxyPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzSxzPre[transferinder1], &DzSxzPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DxSxyPre[transferinder1], &DxSxyPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DySyyPre[transferinder1], &DySyyPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzSyzPre[transferinder1], &DzSyzPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DxSxzPre[transferinder1], &DxSxzPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DySyzPre[transferinder1], &DySyzPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzSzzPre[transferinder1], &DzSzzPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &txx[transferinder1], &txxSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &tyy[transferinder1], &tyySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &txy[transferinder1], &txySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &edxi[transferinder1], &edxiSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edxi2[transferinder1], &edxi2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edyj[transferinder1], &edyjSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edyj2[transferinder1], &edyj2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edzk[transferinder1], &edzkSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edzk2[transferinder1], &edzk2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            while (get_reply != 36)
                                ;
                        }
                    }
                }

                // ------------------------------------------------------两层数值-----------------------------------------------------------------------------------
                if (StepSlaveZ == 0) // 底层的时候取2层
                {
                    for (int conveyy = 0; conveyy < SlaveNyt1; conveyy++)
                    {
                        if ((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3)))
                        {
                        }
                        if (!((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3))))
                        {
                            if (_COL == 7 && (StepSlaveX == limitnxt)) // 对于模型的nxt位置 只取slavenxt1-1个值
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder2UP = (nxt * nyt * (StepSlaveZ + 1)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                int transferinder2DOWN = (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);

                                // 每个从核中 conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex2 = conveyy * SlaveNxt1;
                                // 9A
                                // athread_dma_get(&UPrho_extSlave[getindex2], &rho_ext[transferinder2UP], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&UPtxzSlave[getindex2], &txz[transferinder2UP], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&UPtyzSlave[getindex2], &tyz[transferinder2UP], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&UPtzzSlave[getindex2], &tzz[transferinder2UP], sizeof(float) * (SlaveNxt1 - 2));

                                // athread_dma_get(&DOWNrho_extSlave[getindex2], &rho_ext[transferinder2DOWN], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&DOWNtxzSlave[getindex2], &txz[transferinder2DOWN], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&DOWNtyzSlave[getindex2], &tyz[transferinder2DOWN], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&DOWNtzzSlave[getindex2], &tzz[transferinder2DOWN], sizeof(float) * (SlaveNxt1 - 2));
                                // 5A
                                get_reply = 0;
                                athread_get(PE_MODE, &rho_ext[transferinder2UP], &UPrho_extSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &txz[transferinder2UP], &UPtxzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tyz[transferinder2UP], &UPtyzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tzz[transferinder2UP], &UPtzzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);

                                athread_get(PE_MODE, &rho_ext[transferinder2DOWN], &DOWNrho_extSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &txz[transferinder2DOWN], &DOWNtxzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tyz[transferinder2DOWN], &DOWNtyzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tzz[transferinder2DOWN], &DOWNtzzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                while (get_reply != 8)
                                    ;
                            }
                            if (!(_COL == 7 && (StepSlaveX == limitnxt)))
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder2UP = (nxt * nyt * (StepSlaveZ + 1)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                int transferinder2DOWN = (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex2 = conveyy * SlaveNxt1;
                                // 9A
                                // athread_dma_get(&UPrho_extSlave[getindex2], &rho_ext[transferinder2UP], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&UPtxzSlave[getindex2], &txz[transferinder2UP], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&UPtyzSlave[getindex2], &tyz[transferinder2UP], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&UPtzzSlave[getindex2], &tzz[transferinder2UP], sizeof(float) * SlaveNxt1);

                                // athread_dma_get(&DOWNrho_extSlave[getindex2], &rho_ext[transferinder2DOWN], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&DOWNtxzSlave[getindex2], &txz[transferinder2DOWN], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&DOWNtyzSlave[getindex2], &tyz[transferinder2DOWN], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&DOWNtzzSlave[getindex2], &tzz[transferinder2DOWN], sizeof(float) * SlaveNxt1);

                                // 5A
                                get_reply = 0;
                                athread_get(PE_MODE, &rho_ext[transferinder2UP], &UPrho_extSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &txz[transferinder2UP], &UPtxzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tyz[transferinder2UP], &UPtyzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tzz[transferinder2UP], &UPtzzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);

                                athread_get(PE_MODE, &rho_ext[transferinder2DOWN], &DOWNrho_extSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &txz[transferinder2DOWN], &DOWNtxzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tyz[transferinder2DOWN], &DOWNtyzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tzz[transferinder2DOWN], &DOWNtzzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                while (get_reply != 8)
                                    ;
                            }
                        }
                    }
                }
                if (StepSlaveZ > 0) // 其他地方一次取1层
                {
                    for (int conveyy = 0; conveyy < SlaveNyt1; conveyy++)
                    {
                        if ((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3)))
                        {
                        }
                        if (!((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3))))
                        {
                            if (_COL == 7 && (StepSlaveX == limitnxt)) // 对于模型的nxt位置 只取slavenxt1-1个值
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder2 = (nxt * nyt * (StepSlaveZ + 1)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex2 = conveyy * SlaveNxt1;
                                // 9A
                                // athread_dma_get(&UPrho_extSlave[getindex2], &rho_ext[transferinder2], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&UPtzzSlave[getindex2], &tzz[transferinder2], sizeof(float) * (SlaveNxt1 - 2));

                                // 对于tyz txz 每次取模型中本层放在从核中第二层的值进行计算
                                int transferinder22 = (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                // athread_dma_get(&UPtxzSlave[getindex22], &txz[transferinder22], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&UPtyzSlave[getindex22], &tyz[transferinder22], sizeof(float) * (SlaveNxt1 - 2));
                                int transferinder222 = (nxt * nyt * (StepSlaveZ - 1)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                // 5A
                                get_reply = 0;
                                athread_get(PE_MODE, &rho_ext[transferinder2], &UPrho_extSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tzz[transferinder2], &UPtzzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &rho_ext[transferinder22], &DOWNrho_extSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tzz[transferinder22], &DOWNtzzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);

                                athread_get(PE_MODE, &txz[transferinder22], &UPtxzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tyz[transferinder22], &UPtyzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &txz[transferinder222], &DOWNtxzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tyz[transferinder222], &DOWNtyzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                while (get_reply != 8)
                                    ;
                            }
                            if (!(_COL == 7 && (StepSlaveX == limitnxt)))
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder2 = (nxt * nyt * (StepSlaveZ + 1)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex2 = conveyy * SlaveNxt1;
                                // 9A
                                // athread_dma_get(&UPrho_extSlave[getindex2], &rho_ext[transferinder2], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&UPtzzSlave[getindex2], &tzz[transferinder2], sizeof(float) * SlaveNxt1);

                                // 对于tyz txz 每次取模型中本层放在从核中第二层的值进行计算
                                int transferinder22 = (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                int transferinder222 = (nxt * nyt * (StepSlaveZ - 1)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);

                                // athread_dma_get(&UPtxzSlave[getindex22], &txz[transferinder22], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&UPtyzSlave[getindex22], &tyz[transferinder22], sizeof(float) * SlaveNxt1);

                                // 5A
                                get_reply = 0;
                                athread_get(PE_MODE, &rho_ext[transferinder2], &UPrho_extSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tzz[transferinder2], &UPtzzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &rho_ext[transferinder22], &DOWNrho_extSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tzz[transferinder22], &DOWNtzzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);

                                athread_get(PE_MODE, &txz[transferinder22], &UPtxzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tyz[transferinder22], &UPtyzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &txz[transferinder222], &DOWNtxzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &tyz[transferinder222], &DOWNtyzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                while (get_reply != 8)
                                    ;
                            }
                        }
                    }
                }

                //---------------------------------------------从核计算---------------------------------------------------------------
                // 	//从核内的计算
                if (StepSlaveZ > 0)
                {
                    if ((_COL == 7) && (_ROW == 7) && (StepSlaveY == limitnyt) && (StepSlaveX == limitnxt)) // x边缘处只算8个数和8列数
                    {
                        for (iy = 1; iy < SlaveNyt1 - 3; iy++)
                        {
                            for (ix = 1; ix < SlaveNxt1 - 3; ix++)
                            {
                                // 核内索引
                                // 1层
                                int inderslave1 = iy * SlaveNxt1 + ix;
                                int inderslavex1 = iy * SlaveNxt1 + ix + 1;
                                int inderslavexx1 = iy * SlaveNxt1 + ix - 1;
                                int inderslavey1 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslaveyy1 = (iy - 1) * SlaveNxt1 + ix;

                                // 2层
                                int inderslave2 = iy * SlaveNxt1 + ix;
                                int inderslavex2 = iy * SlaveNxt1 + ix + 1;
                                int inderslavey2 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslavexx2 = iy * SlaveNxt1 + ix - 1;
                                int inderslaveyy2 = (iy - 1) * SlaveNxt1 + ix;

                                rho_tempx = 2 / (DOWNrho_extSlave[inderslave2] + DOWNrho_extSlave[inderslavex2]);
                                rho_tempy = 2 / (DOWNrho_extSlave[inderslave2] + DOWNrho_extSlave[inderslavey2]);
                                rho_tempz = 2 / (DOWNrho_extSlave[inderslave2] + UPrho_extSlave[inderslave2]);

                                // 二阶
                                xx1 = (txxSlave[inderslavex1] - txxSlave[inderslave1]) / H;
                                xy1 = (txySlave[inderslave1] - txySlave[inderslaveyy1]) / H;
                                xz1 = (UPtxzSlave[inderslave2] - DOWNtxzSlave[inderslave2]) / H;

                                yx2 = (txySlave[inderslave1] - txySlave[inderslavexx1]) / H;
                                yy2 = (tyySlave[inderslavey1] - tyySlave[inderslave1]) / H;
                                yz2 = (UPtyzSlave[inderslave2] - DOWNtyzSlave[inderslave2]) / H;

                                zx3 = (UPtxzSlave[inderslave2] - UPtxzSlave[inderslavexx2]) / H;
                                zy3 = (UPtyzSlave[inderslave2] - UPtyzSlave[inderslaveyy2]) / H;
                                zz3 = (UPtzzSlave[inderslave2] - DOWNtzzSlave[inderslave2]) / H;

                                xFHalfTemp2 = -DT * dxi2Slave[inderslave1] * 0.5;
                                xFIntTemp2 = -DT * dxiSlave[inderslave1] * 0.5;
                                yFHalfTemp2 = -DT * dyj2Slave[inderslave1] * 0.5;
                                yFIntTemp2 = -DT * dyjSlave[inderslave1] * 0.5;
                                zFHalfTemp2 = -DT * dzk2Slave[inderslave1] * 0.5;
                                zFIntTemp2 = -DT * dzkSlave[inderslave1] * 0.5;

                                pmlxSxxSlave[inderslave1] = edxi2Slave[inderslave1] * pmlxSxxSlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxSxxPreSlave[inderslave1] + xx1);
                                pmlySxySlave[inderslave1] = edyjSlave[inderslave1] * pmlySxySlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DySxyPreSlave[inderslave1] + xy1);
                                pmlzSxzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzSxzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzSxzPreSlave[inderslave1] + xz1);
                                DxSxxPreSlave[inderslave1] = xx1;
                                DySxyPreSlave[inderslave1] = xy1;
                                DzSxzPreSlave[inderslave1] = xz1;
                                xx1 = xx1 + pmlxSxxSlave[inderslave1];
                                xy1 = xy1 + pmlySxySlave[inderslave1];
                                xz1 = xz1 + pmlzSxzSlave[inderslave1];

                                pmlxSxySlave[inderslave1] = edxiSlave[inderslave1] * pmlxSxySlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxSxyPreSlave[inderslave1] + yx2);
                                pmlySyySlave[inderslave1] = edyj2Slave[inderslave1] * pmlySyySlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DySyyPreSlave[inderslave1] + yy2);
                                pmlzSyzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzSyzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzSyzPreSlave[inderslave1] + yz2);
                                DxSxyPreSlave[inderslave1] = yx2;
                                DySyyPreSlave[inderslave1] = yy2;
                                DzSyzPreSlave[inderslave1] = yz2;
                                yx2 = yx2 + pmlxSxySlave[inderslave1];
                                yy2 = yy2 + pmlySyySlave[inderslave1];
                                yz2 = yz2 + pmlzSyzSlave[inderslave1];

                                pmlxSxzSlave[inderslave1] = edxiSlave[inderslave1] * pmlxSxzSlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxSxzPreSlave[inderslave1] + zx3);
                                pmlySyzSlave[inderslave1] = edyjSlave[inderslave1] * pmlySyzSlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DySyzPreSlave[inderslave1] + zy3);
                                pmlzSzzSlave[inderslave1] = edzk2Slave[inderslave1] * pmlzSzzSlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzSzzPreSlave[inderslave1] + zz3);
                                DxSxzPreSlave[inderslave1] = zx3;
                                DySyzPreSlave[inderslave1] = zy3;
                                DzSzzPreSlave[inderslave1] = zz3;
                                zx3 = zx3 + pmlxSxzSlave[inderslave1];
                                zy3 = zy3 + pmlySyzSlave[inderslave1];
                                zz3 = zz3 + pmlzSzzSlave[inderslave1];

                                vxSlaveresult[inderslave1] = vxSlave[inderslave1] + (DT * rho_tempx) * (xx1 + xy1 + xz1);
                                vySlaveresult[inderslave1] = vySlave[inderslave1] + (DT * rho_tempy) * (yx2 + yy2 + yz2);
                                vzSlaveresult[inderslave1] = vzSlave[inderslave1] + (DT * rho_tempz) * (zx3 + zy3 + zz3);
                            }
                        }
                    }
                    else if (_COL == 7 && (StepSlaveX == limitnxt)) // x边缘处只算8个数
                    {
                        for (iy = 1; iy < SlaveNyt1 - 1; iy++)
                        {
                            for (ix = 1; ix < SlaveNxt1 - 3; ix++)
                            {
                                // 核内索引
                                // 1层
                                int inderslave1 = iy * SlaveNxt1 + ix;
                                int inderslavex1 = iy * SlaveNxt1 + ix + 1;
                                int inderslavexx1 = iy * SlaveNxt1 + ix - 1;
                                int inderslavey1 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslaveyy1 = (iy - 1) * SlaveNxt1 + ix;

                                // 2层
                                int inderslave2 = iy * SlaveNxt1 + ix;
                                int inderslavex2 = iy * SlaveNxt1 + ix + 1;
                                int inderslavey2 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslavexx2 = iy * SlaveNxt1 + ix - 1;
                                int inderslaveyy2 = (iy - 1) * SlaveNxt1 + ix;

                                rho_tempx = 2 / (DOWNrho_extSlave[inderslave2] + DOWNrho_extSlave[inderslavex2]);
                                rho_tempy = 2 / (DOWNrho_extSlave[inderslave2] + DOWNrho_extSlave[inderslavey2]);
                                rho_tempz = 2 / (DOWNrho_extSlave[inderslave2] + UPrho_extSlave[inderslave2]);

                                // 二阶
                                xx1 = (txxSlave[inderslavex1] - txxSlave[inderslave1]) / H;
                                xy1 = (txySlave[inderslave1] - txySlave[inderslaveyy1]) / H;
                                xz1 = (UPtxzSlave[inderslave2] - DOWNtxzSlave[inderslave2]) / H;

                                yx2 = (txySlave[inderslave1] - txySlave[inderslavexx1]) / H;
                                yy2 = (tyySlave[inderslavey1] - tyySlave[inderslave1]) / H;
                                yz2 = (UPtyzSlave[inderslave2] - DOWNtyzSlave[inderslave2]) / H;

                                zx3 = (UPtxzSlave[inderslave2] - UPtxzSlave[inderslavexx2]) / H;
                                zy3 = (UPtyzSlave[inderslave2] - UPtyzSlave[inderslaveyy2]) / H;
                                zz3 = (UPtzzSlave[inderslave2] - DOWNtzzSlave[inderslave2]) / H;

                                xFHalfTemp2 = -DT * dxi2Slave[inderslave1] * 0.5;
                                xFIntTemp2 = -DT * dxiSlave[inderslave1] * 0.5;
                                yFHalfTemp2 = -DT * dyj2Slave[inderslave1] * 0.5;
                                yFIntTemp2 = -DT * dyjSlave[inderslave1] * 0.5;
                                zFHalfTemp2 = -DT * dzk2Slave[inderslave1] * 0.5;
                                zFIntTemp2 = -DT * dzkSlave[inderslave1] * 0.5;

                                pmlxSxxSlave[inderslave1] = edxi2Slave[inderslave1] * pmlxSxxSlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxSxxPreSlave[inderslave1] + xx1);
                                pmlySxySlave[inderslave1] = edyjSlave[inderslave1] * pmlySxySlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DySxyPreSlave[inderslave1] + xy1);
                                pmlzSxzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzSxzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzSxzPreSlave[inderslave1] + xz1);
                                DxSxxPreSlave[inderslave1] = xx1;
                                DySxyPreSlave[inderslave1] = xy1;
                                DzSxzPreSlave[inderslave1] = xz1;
                                xx1 = xx1 + pmlxSxxSlave[inderslave1];
                                xy1 = xy1 + pmlySxySlave[inderslave1];
                                xz1 = xz1 + pmlzSxzSlave[inderslave1];

                                pmlxSxySlave[inderslave1] = edxiSlave[inderslave1] * pmlxSxySlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxSxyPreSlave[inderslave1] + yx2);
                                pmlySyySlave[inderslave1] = edyj2Slave[inderslave1] * pmlySyySlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DySyyPreSlave[inderslave1] + yy2);
                                pmlzSyzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzSyzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzSyzPreSlave[inderslave1] + yz2);
                                DxSxyPreSlave[inderslave1] = yx2;
                                DySyyPreSlave[inderslave1] = yy2;
                                DzSyzPreSlave[inderslave1] = yz2;
                                yx2 = yx2 + pmlxSxySlave[inderslave1];
                                yy2 = yy2 + pmlySyySlave[inderslave1];
                                yz2 = yz2 + pmlzSyzSlave[inderslave1];

                                pmlxSxzSlave[inderslave1] = edxiSlave[inderslave1] * pmlxSxzSlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxSxzPreSlave[inderslave1] + zx3);
                                pmlySyzSlave[inderslave1] = edyjSlave[inderslave1] * pmlySyzSlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DySyzPreSlave[inderslave1] + zy3);
                                pmlzSzzSlave[inderslave1] = edzk2Slave[inderslave1] * pmlzSzzSlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzSzzPreSlave[inderslave1] + zz3);
                                DxSxzPreSlave[inderslave1] = zx3;
                                DySyzPreSlave[inderslave1] = zy3;
                                DzSzzPreSlave[inderslave1] = zz3;
                                zx3 = zx3 + pmlxSxzSlave[inderslave1];
                                zy3 = zy3 + pmlySyzSlave[inderslave1];
                                zz3 = zz3 + pmlzSzzSlave[inderslave1];

                                vxSlaveresult[inderslave1] = vxSlave[inderslave1] + (DT * rho_tempx) * (xx1 + xy1 + xz1);
                                vySlaveresult[inderslave1] = vySlave[inderslave1] + (DT * rho_tempy) * (yx2 + yy2 + yz2);
                                vzSlaveresult[inderslave1] = vzSlave[inderslave1] + (DT * rho_tempz) * (zx3 + zy3 + zz3);
                            }
                        }
                    }
                    else if (_ROW == 7 && (StepSlaveY == limitnyt)) // y边缘处只算8列数
                    {
                        for (iy = 1; iy < SlaveNyt1 - 3; iy++)
                        {
                            for (ix = 1; ix < SlaveNxt1 - 1; ix++)
                            {
                                // 核内索引
                                // 1层
                                int inderslave1 = iy * SlaveNxt1 + ix;
                                int inderslavex1 = iy * SlaveNxt1 + ix + 1;
                                int inderslavexx1 = iy * SlaveNxt1 + ix - 1;
                                int inderslavey1 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslaveyy1 = (iy - 1) * SlaveNxt1 + ix;

                                // 2层
                                int inderslave2 = iy * SlaveNxt1 + ix;
                                int inderslavex2 = iy * SlaveNxt1 + ix + 1;
                                int inderslavey2 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslavexx2 = iy * SlaveNxt1 + ix - 1;
                                int inderslaveyy2 = (iy - 1) * SlaveNxt1 + ix;

                                rho_tempx = 2 / (DOWNrho_extSlave[inderslave2] + DOWNrho_extSlave[inderslavex2]);
                                rho_tempy = 2 / (DOWNrho_extSlave[inderslave2] + DOWNrho_extSlave[inderslavey2]);
                                rho_tempz = 2 / (DOWNrho_extSlave[inderslave2] + UPrho_extSlave[inderslave2]);

                                // 二阶
                                xx1 = (txxSlave[inderslavex1] - txxSlave[inderslave1]) / H;
                                xy1 = (txySlave[inderslave1] - txySlave[inderslaveyy1]) / H;
                                xz1 = (UPtxzSlave[inderslave2] - DOWNtxzSlave[inderslave2]) / H;

                                yx2 = (txySlave[inderslave1] - txySlave[inderslavexx1]) / H;
                                yy2 = (tyySlave[inderslavey1] - tyySlave[inderslave1]) / H;
                                yz2 = (UPtyzSlave[inderslave2] - DOWNtyzSlave[inderslave2]) / H;

                                zx3 = (UPtxzSlave[inderslave2] - UPtxzSlave[inderslavexx2]) / H;
                                zy3 = (UPtyzSlave[inderslave2] - UPtyzSlave[inderslaveyy2]) / H;
                                zz3 = (UPtzzSlave[inderslave2] - DOWNtzzSlave[inderslave2]) / H;

                                xFHalfTemp2 = -DT * dxi2Slave[inderslave1] * 0.5;
                                xFIntTemp2 = -DT * dxiSlave[inderslave1] * 0.5;
                                yFHalfTemp2 = -DT * dyj2Slave[inderslave1] * 0.5;
                                yFIntTemp2 = -DT * dyjSlave[inderslave1] * 0.5;
                                zFHalfTemp2 = -DT * dzk2Slave[inderslave1] * 0.5;
                                zFIntTemp2 = -DT * dzkSlave[inderslave1] * 0.5;

                                pmlxSxxSlave[inderslave1] = edxi2Slave[inderslave1] * pmlxSxxSlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxSxxPreSlave[inderslave1] + xx1);
                                pmlySxySlave[inderslave1] = edyjSlave[inderslave1] * pmlySxySlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DySxyPreSlave[inderslave1] + xy1);
                                pmlzSxzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzSxzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzSxzPreSlave[inderslave1] + xz1);
                                DxSxxPreSlave[inderslave1] = xx1;
                                DySxyPreSlave[inderslave1] = xy1;
                                DzSxzPreSlave[inderslave1] = xz1;
                                xx1 = xx1 + pmlxSxxSlave[inderslave1];
                                xy1 = xy1 + pmlySxySlave[inderslave1];
                                xz1 = xz1 + pmlzSxzSlave[inderslave1];

                                pmlxSxySlave[inderslave1] = edxiSlave[inderslave1] * pmlxSxySlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxSxyPreSlave[inderslave1] + yx2);
                                pmlySyySlave[inderslave1] = edyj2Slave[inderslave1] * pmlySyySlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DySyyPreSlave[inderslave1] + yy2);
                                pmlzSyzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzSyzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzSyzPreSlave[inderslave1] + yz2);
                                DxSxyPreSlave[inderslave1] = yx2;
                                DySyyPreSlave[inderslave1] = yy2;
                                DzSyzPreSlave[inderslave1] = yz2;
                                yx2 = yx2 + pmlxSxySlave[inderslave1];
                                yy2 = yy2 + pmlySyySlave[inderslave1];
                                yz2 = yz2 + pmlzSyzSlave[inderslave1];

                                pmlxSxzSlave[inderslave1] = edxiSlave[inderslave1] * pmlxSxzSlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxSxzPreSlave[inderslave1] + zx3);
                                pmlySyzSlave[inderslave1] = edyjSlave[inderslave1] * pmlySyzSlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DySyzPreSlave[inderslave1] + zy3);
                                pmlzSzzSlave[inderslave1] = edzk2Slave[inderslave1] * pmlzSzzSlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzSzzPreSlave[inderslave1] + zz3);
                                DxSxzPreSlave[inderslave1] = zx3;
                                DySyzPreSlave[inderslave1] = zy3;
                                DzSzzPreSlave[inderslave1] = zz3;
                                zx3 = zx3 + pmlxSxzSlave[inderslave1];
                                zy3 = zy3 + pmlySyzSlave[inderslave1];
                                zz3 = zz3 + pmlzSzzSlave[inderslave1];

                                vxSlaveresult[inderslave1] = vxSlave[inderslave1] + (DT * rho_tempx) * (xx1 + xy1 + xz1);
                                vySlaveresult[inderslave1] = vySlave[inderslave1] + (DT * rho_tempy) * (yx2 + yy2 + yz2);
                                vzSlaveresult[inderslave1] = vzSlave[inderslave1] + (DT * rho_tempz) * (zx3 + zy3 + zz3);
                            }
                        }
                    }
                    if (
                        !((_COL == 7) && (_ROW == 7) && (StepSlaveY == limitnyt) && (StepSlaveX == limitnxt)) &&
                        !((_COL == 7) && (StepSlaveX == limitnxt)) &&
                        !((_ROW == 7) && (StepSlaveY == limitnyt)))
                    {
                        for (iy = 1; iy < SlaveNyt1 - 1; iy++)
                        {
                            for (ix = 1; ix < SlaveNxt1 - 1; ix++)
                            {
                                // 核内索引
                                // 1层
                                int inderslave1 = iy * SlaveNxt1 + ix;
                                int inderslavex1 = iy * SlaveNxt1 + ix + 1;
                                int inderslavexx1 = iy * SlaveNxt1 + ix - 1;
                                int inderslavey1 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslaveyy1 = (iy - 1) * SlaveNxt1 + ix;

                                // 2层
                                int inderslave2 = iy * SlaveNxt1 + ix;
                                int inderslavex2 = iy * SlaveNxt1 + ix + 1;
                                int inderslavey2 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslavexx2 = iy * SlaveNxt1 + ix - 1;
                                int inderslaveyy2 = (iy - 1) * SlaveNxt1 + ix;

                                rho_tempx = 2 / (DOWNrho_extSlave[inderslave2] + DOWNrho_extSlave[inderslavex2]);
                                rho_tempy = 2 / (DOWNrho_extSlave[inderslave2] + DOWNrho_extSlave[inderslavey2]);
                                rho_tempz = 2 / (DOWNrho_extSlave[inderslave2] + UPrho_extSlave[inderslave2]);

                                // 二阶
                                xx1 = (txxSlave[inderslavex1] - txxSlave[inderslave1]) / H;
                                xy1 = (txySlave[inderslave1] - txySlave[inderslaveyy1]) / H;
                                xz1 = (UPtxzSlave[inderslave2] - DOWNtxzSlave[inderslave2]) / H;

                                yx2 = (txySlave[inderslave1] - txySlave[inderslavexx1]) / H;
                                yy2 = (tyySlave[inderslavey1] - tyySlave[inderslave1]) / H;
                                yz2 = (UPtyzSlave[inderslave2] - DOWNtyzSlave[inderslave2]) / H;

                                zx3 = (UPtxzSlave[inderslave2] - UPtxzSlave[inderslavexx2]) / H;
                                zy3 = (UPtyzSlave[inderslave2] - UPtyzSlave[inderslaveyy2]) / H;
                                zz3 = (UPtzzSlave[inderslave2] - DOWNtzzSlave[inderslave2]) / H;

                                xFHalfTemp2 = -DT * dxi2Slave[inderslave1] * 0.5;
                                xFIntTemp2 = -DT * dxiSlave[inderslave1] * 0.5;
                                yFHalfTemp2 = -DT * dyj2Slave[inderslave1] * 0.5;
                                yFIntTemp2 = -DT * dyjSlave[inderslave1] * 0.5;
                                zFHalfTemp2 = -DT * dzk2Slave[inderslave1] * 0.5;
                                zFIntTemp2 = -DT * dzkSlave[inderslave1] * 0.5;

                                pmlxSxxSlave[inderslave1] = edxi2Slave[inderslave1] * pmlxSxxSlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxSxxPreSlave[inderslave1] + xx1);
                                pmlySxySlave[inderslave1] = edyjSlave[inderslave1] * pmlySxySlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DySxyPreSlave[inderslave1] + xy1);
                                pmlzSxzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzSxzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzSxzPreSlave[inderslave1] + xz1);
                                DxSxxPreSlave[inderslave1] = xx1;
                                DySxyPreSlave[inderslave1] = xy1;
                                DzSxzPreSlave[inderslave1] = xz1;
                                xx1 = xx1 + pmlxSxxSlave[inderslave1];
                                xy1 = xy1 + pmlySxySlave[inderslave1];
                                xz1 = xz1 + pmlzSxzSlave[inderslave1];

                                pmlxSxySlave[inderslave1] = edxiSlave[inderslave1] * pmlxSxySlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxSxyPreSlave[inderslave1] + yx2);
                                pmlySyySlave[inderslave1] = edyj2Slave[inderslave1] * pmlySyySlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DySyyPreSlave[inderslave1] + yy2);
                                pmlzSyzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzSyzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzSyzPreSlave[inderslave1] + yz2);
                                DxSxyPreSlave[inderslave1] = yx2;
                                DySyyPreSlave[inderslave1] = yy2;
                                DzSyzPreSlave[inderslave1] = yz2;
                                yx2 = yx2 + pmlxSxySlave[inderslave1];
                                yy2 = yy2 + pmlySyySlave[inderslave1];
                                yz2 = yz2 + pmlzSyzSlave[inderslave1];

                                pmlxSxzSlave[inderslave1] = edxiSlave[inderslave1] * pmlxSxzSlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxSxzPreSlave[inderslave1] + zx3);
                                pmlySyzSlave[inderslave1] = edyjSlave[inderslave1] * pmlySyzSlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DySyzPreSlave[inderslave1] + zy3);
                                pmlzSzzSlave[inderslave1] = edzk2Slave[inderslave1] * pmlzSzzSlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzSzzPreSlave[inderslave1] + zz3);
                                DxSxzPreSlave[inderslave1] = zx3;
                                DySyzPreSlave[inderslave1] = zy3;
                                DzSzzPreSlave[inderslave1] = zz3;
                                zx3 = zx3 + pmlxSxzSlave[inderslave1];
                                zy3 = zy3 + pmlySyzSlave[inderslave1];
                                zz3 = zz3 + pmlzSzzSlave[inderslave1];

                                vxSlaveresult[inderslave1] = vxSlave[inderslave1] + (DT * rho_tempx) * (xx1 + xy1 + xz1);
                                vySlaveresult[inderslave1] = vySlave[inderslave1] + (DT * rho_tempy) * (yx2 + yy2 + yz2);
                                vzSlaveresult[inderslave1] = vzSlave[inderslave1] + (DT * rho_tempz) * (zx3 + zy3 + zz3);
                            }
                        }
                    }
                }

                //------------------------------------------回传-----------------------------------------------------------
                // 向主核中传数组DMA传输
                if (StepSlaveZ > 0)
                {
                    for (int conveyy = 1; conveyy < (SlaveNyt1 - 1); conveyy++)
                    {

                        if ((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3)))
                        {
                        }
                        if (!((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3))))
                        {

                            if (_COL == 7 && (StepSlaveX == limitnxt)) // x在边界 每列只传8个数
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                // 回传所以需要从每个从核中的第二列开始算也就是x方向+1的位置开始取10个数
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt) + 1;
                                // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt1 + 1;
                                // 9A
                                // athread_dma_put(&vx[transferinder], &vxSlaveresult[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&vy[transferinder], &vySlaveresult[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&vz[transferinder], &vzSlaveresult[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DxSxxPre[transferinder], &DxSxxPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DySxyPre[transferinder], &DySxyPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DzSxzPre[transferinder], &DzSxzPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DxSxyPre[transferinder], &DxSxyPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DySyyPre[transferinder], &DySyyPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DzSyzPre[transferinder], &DzSyzPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DxSxzPre[transferinder], &DxSxzPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DySyzPre[transferinder], &DySyzPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DzSzzPre[transferinder], &DzSzzPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlxSxx[transferinder], &pmlxSxxSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlySxy[transferinder], &pmlySxySlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlzSxz[transferinder], &pmlzSxzSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlxSxy[transferinder], &pmlxSxySlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlySyy[transferinder], &pmlySyySlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlzSyz[transferinder], &pmlzSyzSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlxSxz[transferinder], &pmlxSxzSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlySyz[transferinder], &pmlySyzSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlzSzz[transferinder], &pmlzSzzSlave[getindex], sizeof(float) * (SlaveNxt - 2));

                                // 5A
                                put_reply = 0;
                                athread_put(PE_MODE, &vxSlaveresult[getindex], &vx[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &vySlaveresult[getindex], &vy[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &vzSlaveresult[getindex], &vz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxSxxPreSlave[getindex], &DxSxxPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DySxyPreSlave[getindex], &DySxyPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzSxzPreSlave[getindex], &DzSxzPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxSxyPreSlave[getindex], &DxSxyPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DySyyPreSlave[getindex], &DySyyPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzSyzPreSlave[getindex], &DzSyzPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxSxzPreSlave[getindex], &DxSxzPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DySyzPreSlave[getindex], &DySyzPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzSzzPreSlave[getindex], &DzSzzPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxSxxSlave[getindex], &pmlxSxx[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlySxySlave[getindex], &pmlySxy[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzSxzSlave[getindex], &pmlzSxz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxSxySlave[getindex], &pmlxSxy[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlySyySlave[getindex], &pmlySyy[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzSyzSlave[getindex], &pmlzSyz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxSxzSlave[getindex], &pmlxSxz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlySyzSlave[getindex], &pmlySyz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzSzzSlave[getindex], &pmlzSzz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                while (put_reply != 21)
                                    ;
                            }
                            if (!(_COL == 7 && (StepSlaveX == limitnxt)))
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                // 回传所以需要从每个从核中的第二列开始算也就是x方向+1的位置开始取10个数
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt) + 1;
                                // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt1 + 1;
                                // 9A
                                // athread_dma_put(&vx[transferinder], &vxSlaveresult[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&vy[transferinder], &vySlaveresult[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&vz[transferinder], &vzSlaveresult[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DxSxxPre[transferinder], &DxSxxPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DySxyPre[transferinder], &DySxyPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DzSxzPre[transferinder], &DzSxzPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DxSxyPre[transferinder], &DxSxyPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DySyyPre[transferinder], &DySyyPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DzSyzPre[transferinder], &DzSyzPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DxSxzPre[transferinder], &DxSxzPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DySyzPre[transferinder], &DySyzPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DzSzzPre[transferinder], &DzSzzPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlxSxx[transferinder], &pmlxSxxSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlySxy[transferinder], &pmlySxySlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlzSxz[transferinder], &pmlzSxzSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlxSxy[transferinder], &pmlxSxySlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlySyy[transferinder], &pmlySyySlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlzSyz[transferinder], &pmlzSyzSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlxSxz[transferinder], &pmlxSxzSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlySyz[transferinder], &pmlySyzSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlzSzz[transferinder], &pmlzSzzSlave[getindex], sizeof(float) * SlaveNxt);

                                // 5A
                                put_reply = 0;
                                athread_put(PE_MODE, &vxSlaveresult[getindex], &vx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &vySlaveresult[getindex], &vy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &vzSlaveresult[getindex], &vz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxSxxPreSlave[getindex], &DxSxxPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DySxyPreSlave[getindex], &DySxyPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzSxzPreSlave[getindex], &DzSxzPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxSxyPreSlave[getindex], &DxSxyPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DySyyPreSlave[getindex], &DySyyPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzSyzPreSlave[getindex], &DzSyzPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxSxzPreSlave[getindex], &DxSxzPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DySyzPreSlave[getindex], &DySyzPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzSzzPreSlave[getindex], &DzSzzPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxSxxSlave[getindex], &pmlxSxx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlySxySlave[getindex], &pmlySxy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzSxzSlave[getindex], &pmlzSxz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxSxySlave[getindex], &pmlxSxy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlySyySlave[getindex], &pmlySyy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzSyzSlave[getindex], &pmlzSyz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxSxzSlave[getindex], &pmlxSxz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlySyzSlave[getindex], &pmlySyz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzSzzSlave[getindex], &pmlzSzz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 21)
                                    ;
                            }
                        }
                    }
                }
                /* //---------------------------------------------从核内重新赋值--------------------------------------------------------------
                // 当 stepSlaveZ == 0
                if (StepSlaveZ == 0)
                {
                    // 每次取上层的值 放到下层
                    MIDDLErho_extSlave = DOWNrho_extSlave;
                    DOWNrho_extSlave = UPrho_extSlave;
                    UPrho_extSlave = MIDDLErho_extSlave;

                    MIDDLEtzzSlave = DOWNtzzSlave;
                    DOWNtzzSlave = UPtzzSlave;
                    UPtzzSlave = MIDDLEtzzSlave;
                }
                // 二层数值的参数循环幅值 将最上层值传到中间，中间值传到底面
                if (StepSlaveZ > 0)
                {
                    // 每次取上层的值 放到下层
                    MIDDLErho_extSlave = DOWNrho_extSlave;
                    DOWNrho_extSlave = UPrho_extSlave;
                    UPrho_extSlave = MIDDLErho_extSlave;

                    MIDDLEtzzSlave = DOWNtzzSlave;
                    DOWNtzzSlave = UPtzzSlave;
                    UPtzzSlave = MIDDLEtzzSlave;

                    MIDDLEtxzSlave = DOWNtxzSlave;
                    DOWNtxzSlave = UPtxzSlave;
                    UPtxzSlave = MIDDLEtxzSlave;

                    MIDDLEtyzSlave = DOWNtyzSlave;
                    DOWNtyzSlave = UPtyzSlave;
                    UPtyzSlave = MIDDLEtyzSlave;
                } */
            }
        }
    }
    // // if(myid == 4 && it == 10)
    // // {
    // lwpf_stop(A);
    // //}
    // lwpf_exit(test);
}
//**********************************************************************************************************************************************************************
//                                                          计算应力
//**********************************************************************************************************************************************************************
void funStress()
{
    float y5, y6, y7, y8, y9, y10, y11, y12, y13;
    // 单个从核中的计算长度
    int SlaveNxt, SlaveNyt, SlaveNzt, slaveAll;
    SlaveNxt = 10;
    SlaveNyt = 10;
    SlaveNzt = 1;
    slaveAll = SlaveNxt * SlaveNyt * SlaveNzt;
    // 单个从核中的计算长度多传的外壳
    int SlaveNxt1, SlaveNyt1, SlaveNzt1, slaveAll1, slaveAll2, slaveAll3;
    SlaveNxt1 = SlaveNxt + 2;
    SlaveNyt1 = SlaveNyt + 2;
    SlaveNzt1 = SlaveNzt + 2;
    slaveAll3 = SlaveNxt1 * SlaveNyt1 * SlaveNzt1;
    slaveAll2 = SlaveNxt1 * SlaveNyt1 * 2;
    slaveAll1 = SlaveNxt1 * SlaveNyt1;
    // x y z 方向上的从核数
    int slaveXnumb = 8;
    int slaveYnumb = 8;
    int slaveZnumb = 1;
    // 开辟空间
    float pmlxVxSlave[slaveAll1], pmlyVySlave[slaveAll1], pmlzVzSlave[slaveAll1], pmlzVxSlave[slaveAll1], pmlxVzSlave[slaveAll1], pmlyVxSlave[slaveAll1], pmlxVySlave[slaveAll1], pmlzVySlave[slaveAll1], pmlyVzSlave[slaveAll1];
    float DxVxPreSlave[slaveAll1], DyVyPreSlave[slaveAll1], DzVzPreSlave[slaveAll1], DzVxPreSlave[slaveAll1], DxVzPreSlave[slaveAll1], DyVxPreSlave[slaveAll1], DxVyPreSlave[slaveAll1], DzVyPreSlave[slaveAll1], DyVzPreSlave[slaveAll1];
    float txxSlave[slaveAll1], tyySlave[slaveAll1], tzzSlave[slaveAll1], txySlave[slaveAll1], txzSlave[slaveAll1], tyzSlave[slaveAll1], dxiSlave[slaveAll1], dxi2Slave[slaveAll1], dyjSlave[slaveAll1], dyj2Slave[slaveAll1], dzkSlave[slaveAll1], dzk2Slave[slaveAll1], lamda2u_extSlave[slaveAll1], lamda_extSlave[slaveAll1];
    float txxSlaveresult[slaveAll1], tyySlaveresult[slaveAll1], tzzSlaveresult[slaveAll1], txySlaveresult[slaveAll1], txzSlaveresult[slaveAll1], tyzSlaveresult[slaveAll1];
    float muxySlave[slaveAll1], muxzSlave[slaveAll1], muyzSlave[slaveAll1];
    float edxiSlave[slaveAll1], edxi2Slave[slaveAll1], edyjSlave[slaveAll1], edyj2Slave[slaveAll1], edzkSlave[slaveAll1], edzk2Slave[slaveAll1];
    // 二层
    float vxSlaveUp[slaveAll1], vySlaveUp[slaveAll1], vzSlaveUp[slaveAll1];
    float vxSlaveDown[slaveAll1], vySlaveDown[slaveAll1], vzSlaveDown[slaveAll1];
    // 进行掩盖的中间变量
    float *UPvxSlave, *DOWNvxSlave, *MIDDLEvxSlave;
    float *UPvySlave, *DOWNvySlave, *MIDDLEvySlave;
    float *UPvzSlave, *DOWNvzSlave, *MIDDLEvzSlave;
    // 开辟空间
    // 一层
    memset(pmlxVxSlave, 0, slaveAll1 * sizeof(float));
    memset(pmlyVySlave, 0, slaveAll1 * sizeof(float));
    memset(pmlzVzSlave, 0, slaveAll1 * sizeof(float));
    memset(pmlzVxSlave, 0, slaveAll1 * sizeof(float));
    memset(pmlxVzSlave, 0, slaveAll1 * sizeof(float));
    memset(pmlyVxSlave, 0, slaveAll1 * sizeof(float));
    memset(pmlxVySlave, 0, slaveAll1 * sizeof(float));
    memset(pmlzVySlave, 0, slaveAll1 * sizeof(float));
    memset(pmlyVzSlave, 0, slaveAll1 * sizeof(float));
    memset(DxVxPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DyVyPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DzVzPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DzVxPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DxVzPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DyVxPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DxVyPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DzVyPreSlave, 0, slaveAll1 * sizeof(float));
    memset(DyVzPreSlave, 0, slaveAll1 * sizeof(float));
    memset(txxSlave, 0, slaveAll1 * sizeof(float));
    memset(tyySlave, 0, slaveAll1 * sizeof(float));
    memset(tzzSlave, 0, slaveAll1 * sizeof(float));
    memset(txySlave, 0, slaveAll1 * sizeof(float));
    memset(txzSlave, 0, slaveAll1 * sizeof(float));
    memset(tyzSlave, 0, slaveAll1 * sizeof(float));
    memset(dxiSlave, 0, slaveAll1 * sizeof(float));
    memset(dxi2Slave, 0, slaveAll1 * sizeof(float));
    memset(dyjSlave, 0, slaveAll1 * sizeof(float));
    memset(dyj2Slave, 0, slaveAll1 * sizeof(float));
    memset(dzkSlave, 0, slaveAll1 * sizeof(float));
    memset(dzk2Slave, 0, slaveAll1 * sizeof(float));
    memset(edxiSlave, 0, slaveAll1 * sizeof(float));
    memset(edxi2Slave, 0, slaveAll1 * sizeof(float));
    memset(edyjSlave, 0, slaveAll1 * sizeof(float));
    memset(edyj2Slave, 0, slaveAll1 * sizeof(float));
    memset(edzkSlave, 0, slaveAll1 * sizeof(float));
    memset(edzk2Slave, 0, slaveAll1 * sizeof(float));
    memset(muxzSlave, 0, slaveAll1 * sizeof(float));
    memset(muxySlave, 0, slaveAll1 * sizeof(float));
    memset(muyzSlave, 0, slaveAll1 * sizeof(float));
    memset(lamda2u_extSlave, 0, slaveAll1 * sizeof(float));
    memset(lamda_extSlave, 0, slaveAll1 * sizeof(float));
    memset(txxSlaveresult, 0, slaveAll1 * sizeof(float));
    memset(tyySlaveresult, 0, slaveAll1 * sizeof(float));
    memset(tzzSlaveresult, 0, slaveAll1 * sizeof(float));
    memset(txySlaveresult, 0, slaveAll1 * sizeof(float));
    memset(txzSlaveresult, 0, slaveAll1 * sizeof(float));
    memset(tyzSlaveresult, 0, slaveAll1 * sizeof(float));
    // 二层
    memset(vxSlaveUp, 0, slaveAll1 * sizeof(float));
    memset(vySlaveUp, 0, slaveAll1 * sizeof(float));
    memset(vzSlaveUp, 0, slaveAll1 * sizeof(float));
    memset(vxSlaveDown, 0, slaveAll1 * sizeof(float));
    memset(vySlaveDown, 0, slaveAll1 * sizeof(float));
    memset(vzSlaveDown, 0, slaveAll1 * sizeof(float));
    // 进行赋值掩盖
    UPvxSlave = vxSlaveUp;
    DOWNvxSlave = vxSlaveDown;
    UPvySlave = vySlaveUp;
    DOWNvySlave = vySlaveDown;
    UPvzSlave = vzSlaveUp;
    DOWNvzSlave = vzSlaveDown;
    int limitnxt, limitnyt;
    limitnxt = ceil((nxt / (SlaveNxt * slaveXnumb)) - 1) * (SlaveNxt * slaveXnumb);
    limitnyt = ceil((nyt / (SlaveNyt * slaveYnumb)) - 1) * (SlaveNyt * slaveYnumb);
    // 计算应力
    // 进行从核的跳步
    for (int StepSlaveZ = 0; StepSlaveZ < (nzt - 1); StepSlaveZ++)
    {
        for (int StepSlaveY = 0; StepSlaveY < nyt; StepSlaveY += (SlaveNyt * slaveYnumb))
        {
            for (int StepSlaveX = 0; StepSlaveX < nxt; StepSlaveX += (SlaveNxt * slaveXnumb))
            {
                //------------------------------------------读数据--------------------------------------------------------------
                // 从主核中取数组DMA传输   每次从主核多读一行，在主核中表现为每个从核间取值有交叉
                // 一层  从iz=1开始取 iz=0不用取
                for (int conveyy = 0; conveyy < SlaveNyt1; conveyy++)
                {
                    if ((_ROW == 7) && (conveyy > (SlaveNyt1 - 3)) && (StepSlaveY == limitnyt))
                    {
                    }
                    if (!((_ROW == 7) && (conveyy > (SlaveNyt1 - 3)) && (StepSlaveY == limitnyt)))
                    {
                        // 对于模型的nxt位置 只取slavenxt1-1个值
                        if (_COL == 7 && (StepSlaveX == limitnxt))
                        {
                            //-----------------------------------------------传输数据DMA对应在模型上的坐标索引----------------------------------------------------------------
                            // 本层取值
                            // 传输数据DMA对应在模型上的坐标索引
                            // (nxt * nyt * (StepSlaveZ)) 到对应的那一层的首地址  在z方向上
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder1 = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex1 = conveyy * SlaveNxt1;

                            // 9A
                            // athread_dma_get(&txxSlave[getindex1], &txx[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&tyySlave[getindex1], &tyy[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&tzzSlave[getindex1], &tzz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&txySlave[getindex1], &txy[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&txzSlave[getindex1], &txz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&tyzSlave[getindex1], &tyz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dxiSlave[getindex1], &dxi[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dxi2Slave[getindex1], &dxi2[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dyjSlave[getindex1], &dyj[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dyj2Slave[getindex1], &dyj2[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dzkSlave[getindex1], &dzk[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&dzk2Slave[getindex1], &dzk2[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&lamda2u_extSlave[getindex1], &lamda2u_ext[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&lamda_extSlave[getindex1], &lamda_ext[transferinder1], sizeof(float) * (SlaveNxt1 - 2));

                            // athread_dma_get(&pmlxVxSlave[getindex1], &pmlxVx[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlyVySlave[getindex1], &pmlyVy[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlzVzSlave[getindex1], &pmlzVz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlzVxSlave[getindex1], &pmlzVx[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlxVzSlave[getindex1], &pmlxVz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlyVxSlave[getindex1], &pmlyVx[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlxVySlave[getindex1], &pmlxVy[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlzVySlave[getindex1], &pmlzVy[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&pmlyVzSlave[getindex1], &pmlyVz[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DxVxPreSlave[getindex1], &DxVxPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DyVyPreSlave[getindex1], &DyVyPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DzVzPreSlave[getindex1], &DzVzPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DzVxPreSlave[getindex1], &DzVxPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DxVzPreSlave[getindex1], &DxVzPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DyVxPreSlave[getindex1], &DyVxPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DxVyPreSlave[getindex1], &DxVyPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DzVyPreSlave[getindex1], &DzVyPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));
                            // athread_dma_get(&DyVzPreSlave[getindex1], &DyVzPre[transferinder1], sizeof(float) * (SlaveNxt1 - 2));

                            // 5A
                            get_reply = 0;
                            athread_get(PE_MODE, &txx[transferinder1], &txxSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &tyy[transferinder1], &tyySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &tzz[transferinder1], &tzzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &txy[transferinder1], &txySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &txz[transferinder1], &txzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &tyz[transferinder1], &tyzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dxi[transferinder1], &dxiSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dxi2[transferinder1], &dxi2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dyj[transferinder1], &dyjSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dyj2[transferinder1], &dyj2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dzk[transferinder1], &dzkSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dzk2[transferinder1], &dzk2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &lamda2u_ext[transferinder1], &lamda2u_extSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &lamda_ext[transferinder1], &lamda_extSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &edxi[transferinder1], &edxiSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edxi2[transferinder1], &edxi2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edyj[transferinder1], &edyjSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edyj2[transferinder1], &edyj2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edzk[transferinder1], &edzkSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edzk2[transferinder1], &edzk2Slave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &muxy[transferinder1], &muxySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &muxz[transferinder1], &muxzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &muyz[transferinder1], &muyzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &pmlxVx[transferinder1], &pmlxVxSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlyVy[transferinder1], &pmlyVySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzVz[transferinder1], &pmlzVzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzVx[transferinder1], &pmlzVxSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlxVz[transferinder1], &pmlxVzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlyVx[transferinder1], &pmlyVxSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlxVy[transferinder1], &pmlxVySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzVy[transferinder1], &pmlzVySlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlyVz[transferinder1], &pmlyVzSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DxVxPre[transferinder1], &DxVxPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DyVyPre[transferinder1], &DyVyPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzVzPre[transferinder1], &DzVzPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzVxPre[transferinder1], &DzVxPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DxVzPre[transferinder1], &DxVzPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DyVxPre[transferinder1], &DyVxPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DxVyPre[transferinder1], &DxVyPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzVyPre[transferinder1], &DzVyPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DyVzPre[transferinder1], &DyVzPreSlave[getindex1], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                            while (get_reply != 41)
                                ;
                        }
                        if (!(_COL == 7 && (StepSlaveX == limitnxt)))
                        {
                            // 传输数据DMA对应在模型上的坐标索引
                            // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                            // (StepSlaveX) 到索引所在的核组的首地址
                            // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                            // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                            // (conveyy * nxt) 到索引所在的列了
                            int transferinder1 = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                            // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                            int getindex1 = conveyy * SlaveNxt1;
                            // 9A
                            // athread_dma_get(&txxSlave[getindex1], &txx[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&tyySlave[getindex1], &tyy[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&tzzSlave[getindex1], &tzz[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&txySlave[getindex1], &txy[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&txzSlave[getindex1], &txz[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&tyzSlave[getindex1], &tyz[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dxiSlave[getindex1], &dxi[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dxi2Slave[getindex1], &dxi2[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dyjSlave[getindex1], &dyj[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dyj2Slave[getindex1], &dyj2[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dzkSlave[getindex1], &dzk[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&dzk2Slave[getindex1], &dzk2[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&lamda2u_extSlave[getindex1], &lamda2u_ext[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&lamda_extSlave[getindex1], &lamda_ext[transferinder1], sizeof(float) * SlaveNxt1);

                            // athread_dma_get(&DxVxPreSlave[getindex1], &DxVxPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DyVyPreSlave[getindex1], &DyVyPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DzVzPreSlave[getindex1], &DzVzPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DzVxPreSlave[getindex1], &DzVxPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DxVzPreSlave[getindex1], &DxVzPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DyVxPreSlave[getindex1], &DyVxPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DxVyPreSlave[getindex1], &DxVyPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DzVyPreSlave[getindex1], &DzVyPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&DyVzPreSlave[getindex1], &DyVzPre[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlxVxSlave[getindex1], &pmlxVx[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlyVySlave[getindex1], &pmlyVy[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlzVzSlave[getindex1], &pmlzVz[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlzVxSlave[getindex1], &pmlzVx[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlxVzSlave[getindex1], &pmlxVz[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlyVxSlave[getindex1], &pmlyVx[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlxVySlave[getindex1], &pmlxVy[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlzVySlave[getindex1], &pmlzVy[transferinder1], sizeof(float) * SlaveNxt1);
                            // athread_dma_get(&pmlyVzSlave[getindex1], &pmlyVz[transferinder1], sizeof(float) * SlaveNxt1);

                            // 5A
                            get_reply = 0;
                            athread_get(PE_MODE, &txx[transferinder1], &txxSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &tyy[transferinder1], &tyySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &tzz[transferinder1], &tzzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &txy[transferinder1], &txySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &txz[transferinder1], &txzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &tyz[transferinder1], &tyzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dxi[transferinder1], &dxiSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dxi2[transferinder1], &dxi2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dyj[transferinder1], &dyjSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dyj2[transferinder1], &dyj2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dzk[transferinder1], &dzkSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &dzk2[transferinder1], &dzk2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &lamda2u_ext[transferinder1], &lamda2u_extSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &lamda_ext[transferinder1], &lamda_extSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &edxi[transferinder1], &edxiSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edxi2[transferinder1], &edxi2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edyj[transferinder1], &edyjSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edyj2[transferinder1], &edyj2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edzk[transferinder1], &edzkSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &edzk2[transferinder1], &edzk2Slave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &muxy[transferinder1], &muxySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &muyz[transferinder1], &muyzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &muxz[transferinder1], &muxzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);

                            athread_get(PE_MODE, &DxVxPre[transferinder1], &DxVxPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DyVyPre[transferinder1], &DyVyPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzVzPre[transferinder1], &DzVzPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzVxPre[transferinder1], &DzVxPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DxVzPre[transferinder1], &DxVzPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DyVxPre[transferinder1], &DyVxPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DxVyPre[transferinder1], &DxVyPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DzVyPre[transferinder1], &DzVyPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &DyVzPre[transferinder1], &DyVzPreSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlxVx[transferinder1], &pmlxVxSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlyVy[transferinder1], &pmlyVySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzVz[transferinder1], &pmlzVzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzVx[transferinder1], &pmlzVxSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlxVz[transferinder1], &pmlxVzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlyVx[transferinder1], &pmlyVxSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlxVy[transferinder1], &pmlxVySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlzVy[transferinder1], &pmlzVySlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            athread_get(PE_MODE, &pmlyVz[transferinder1], &pmlyVzSlave[getindex1], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                            while (get_reply != 41)
                                ;
                        }
                    }
                }

                // ------------------------------------------------------两层数值-----------------------------------------------------------------------------------
                if (StepSlaveZ == 0) // 底层的时候取2层
                {
                    for (int conveyy = 0; conveyy < SlaveNyt1; conveyy++)
                    {
                        if ((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3)))
                        {
                        }
                        if (!((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3))))
                        {
                            if (_COL == 7 && (StepSlaveX == limitnxt)) // 对于模型的nxt位置 只取slavenxt1-1个值
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder2UP = (nxt * nyt * (StepSlaveZ + 1)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                int transferinder2DOWN = (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex2 = conveyy * SlaveNxt1;
                                // 9A
                                // athread_dma_get(&UPmu_extSlave[getindex2], &mu_ext[transferinder2UP], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&UPvxSlave[getindex2], &vx[transferinder2UP], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&UPvySlave[getindex2], &vy[transferinder2UP], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&UPvzSlave[getindex2], &vz[transferinder2UP], sizeof(float) * (SlaveNxt1 - 2));

                                // athread_dma_get(&DOWNmu_extSlave[getindex2], &mu_ext[transferinder2DOWN], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&DOWNvxSlave[getindex2], &vx[transferinder2DOWN], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&DOWNvySlave[getindex2], &vy[transferinder2DOWN], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&DOWNvzSlave[getindex2], &vz[transferinder2DOWN], sizeof(float) * (SlaveNxt1 - 2));

                                // 5A
                                get_reply = 0;

                                athread_get(PE_MODE, &vx[transferinder2UP], &UPvxSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vy[transferinder2UP], &UPvySlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vz[transferinder2UP], &UPvzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);

                                athread_get(PE_MODE, &vx[transferinder2DOWN], &DOWNvxSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vy[transferinder2DOWN], &DOWNvySlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vz[transferinder2DOWN], &DOWNvzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                while (get_reply != 6)
                                    ;
                            }
                            if (!(_COL == 7 && (StepSlaveX == limitnxt)))
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder2UP = (nxt * nyt * (StepSlaveZ + 1)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                int transferinder2DOWN = (nxt * nyt * (StepSlaveZ)) + (StepSlaveY + _ROW * SlaveNyt + conveyy) * nxt + (StepSlaveX + _COL * SlaveNxt);
                                // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex2 = conveyy * SlaveNxt1;
                                // 9A
                                // athread_dma_get(&UPmu_extSlave[getindex2], &mu_ext[transferinder2UP], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&UPvxSlave[getindex2], &vx[transferinder2UP], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&UPvySlave[getindex2], &vy[transferinder2UP], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&UPvzSlave[getindex2], &vz[transferinder2UP], sizeof(float) * SlaveNxt1);

                                // athread_dma_get(&DOWNmu_extSlave[getindex2], &mu_ext[transferinder2DOWN], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&DOWNvxSlave[getindex2], &vx[transferinder2DOWN], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&DOWNvySlave[getindex2], &vy[transferinder2DOWN], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&DOWNvzSlave[getindex2], &vz[transferinder2DOWN], sizeof(float) * SlaveNxt1);

                                // 5A
                                get_reply = 0;

                                athread_get(PE_MODE, &vx[transferinder2UP], &UPvxSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vy[transferinder2UP], &UPvySlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vz[transferinder2UP], &UPvzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);

                                athread_get(PE_MODE, &vx[transferinder2DOWN], &DOWNvxSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vy[transferinder2DOWN], &DOWNvySlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vz[transferinder2DOWN], &DOWNvzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                while (get_reply != 6)
                                    ;
                            }
                        }
                    }
                }
                if (StepSlaveZ > 0) // 其他地方一次取1层
                {
                    for (int conveyy = 0; conveyy < SlaveNyt1; conveyy++)
                    {

                        if ((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3)))
                        {
                        }
                        if (!((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3))))
                        {
                            if (_COL == 7 && (StepSlaveX == limitnxt)) // 对于模型的nxt位置 只取slavenxt1-1个值
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder2 = (nxt * nyt * (StepSlaveZ + 1)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex2 = conveyy * SlaveNxt1;
                                int transferinder222 = (nxt * nyt * (StepSlaveZ - 1)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // athread_dma_get(&UPmu_extSlave[getindex2], &mu_ext[transferinder2], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&UPvxSlave[getindex2], &vx[transferinder2], sizeof(float) * (SlaveNxt1 - 2));
                                // athread_dma_get(&UPvySlave[getindex2], &vy[transferinder2], sizeof(float) * (SlaveNxt1 - 2));

                                // 对于tyz txz 每次取模型中本层放在从核中第二层的值进行计算
                                int transferinder22 = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // athread_dma_get(&UPvzSlave[getindex22], &vz[transferinder22], sizeof(float) * (SlaveNxt1 - 2));

                                // 5A
                                get_reply = 0;
                                athread_get(PE_MODE, &vx[transferinder2], &UPvxSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vy[transferinder2], &UPvySlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vx[transferinder22], &DOWNvxSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vy[transferinder22], &DOWNvySlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);

                                athread_get(PE_MODE, &vz[transferinder22], &UPvzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vz[transferinder222], &DOWNvzSlave[getindex2], sizeof(float) * (SlaveNxt1 - 2), &get_reply, 0, 0, 0);
                                while (get_reply != 6)
                                    ;
                            }
                            if (!(_COL == 7 && (StepSlaveX == limitnxt)))
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                int transferinder2 = (nxt * nyt * (StepSlaveZ + 1)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex2 = conveyy * SlaveNxt1;
                                int transferinder222 = (nxt * nyt * (StepSlaveZ - 1)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // athread_dma_get(&UPmu_extSlave[getindex2], &mu_ext[transferinder2], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&UPvxSlave[getindex2], &vx[transferinder2], sizeof(float) * SlaveNxt1);
                                // athread_dma_get(&UPvySlave[getindex2], &vy[transferinder2], sizeof(float) * SlaveNxt1);

                                // 对于tyz txz 每次取模型中本层放在从核中第二层的值进行计算
                                int transferinder22 = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt);
                                // athread_dma_get(&UPvzSlave[getindex22], &vz[transferinder22], sizeof(float) * SlaveNxt1);

                                // 5A
                                get_reply = 0;
                                athread_get(PE_MODE, &vx[transferinder2], &UPvxSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vy[transferinder2], &UPvySlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vx[transferinder22], &DOWNvxSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vy[transferinder22], &DOWNvySlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);

                                athread_get(PE_MODE, &vz[transferinder22], &UPvzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                athread_get(PE_MODE, &vz[transferinder222], &DOWNvzSlave[getindex2], sizeof(float) * SlaveNxt1, &get_reply, 0, 0, 0);
                                while (get_reply != 6)
                                    ;
                            }
                        }
                    }
                }

                //----------------------------------------------计算--------------------------------------------------------------
                // 	//从核内的计算
                if (StepSlaveZ > 0)
                {
                    if ((_COL == 7) && (_ROW == 7) && (StepSlaveY == limitnyt) && (StepSlaveX == limitnxt))
                    {
                        for (iy = 1; iy < SlaveNyt1 - 3; iy++)
                        {
                            for (ix = 1; ix < SlaveNxt1 - 3; ix++)
                            {

                                // 核内索引
                                // 1层
                                int inderslave1 = iy * SlaveNxt1 + ix;

                                // 2层
                                int inderslave2 = iy * SlaveNxt1 + ix;
                                int inderslavex2 = iy * SlaveNxt1 + ix + 1;
                                int inderslavexx2 = iy * SlaveNxt1 + ix - 1;
                                int inderslavey2 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslaveyy2 = (iy - 1) * SlaveNxt1 + ix;
                                int inderslavexy2 = (iy + 1) * SlaveNxt1 + ix + 1;

                                y5 = (DOWNvxSlave[inderslave2] - DOWNvxSlave[inderslavexx2]) / H;
                                y6 = (DOWNvySlave[inderslave2] - DOWNvySlave[inderslaveyy2]) / H;
                                y7 = (UPvzSlave[inderslave2] - DOWNvzSlave[inderslave2]) / H;
                                y8 = (DOWNvxSlave[inderslavey2] - DOWNvxSlave[inderslave2]) / H;
                                y9 = (DOWNvySlave[inderslavex2] - DOWNvySlave[inderslave2]) / H;
                                y10 = (UPvxSlave[inderslave2] - DOWNvxSlave[inderslave2]) / H;
                                y11 = (UPvzSlave[inderslavex2] - UPvzSlave[inderslave2]) / H;
                                y12 = (UPvySlave[inderslave2] - DOWNvySlave[inderslave2]) / H;
                                y13 = (UPvzSlave[inderslavey2] - UPvzSlave[inderslave2]) / H;

                                xFHalfTemp2 = -DT * dxi2Slave[inderslave1] * 0.5;
                                xFIntTemp2 = -DT * dxiSlave[inderslave1] * 0.5;
                                yFHalfTemp2 = -DT * dyj2Slave[inderslave1] * 0.5;
                                yFIntTemp2 = -DT * dyjSlave[inderslave1] * 0.5;
                                zFHalfTemp2 = -DT * dzk2Slave[inderslave1] * 0.5;
                                zFIntTemp2 = -DT * dzkSlave[inderslave1] * 0.5;
                                pmlxVxSlave[inderslave1] = edxiSlave[inderslave1] * pmlxVxSlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxVxPreSlave[inderslave1] + y5);
                                pmlyVySlave[inderslave1] = edyjSlave[inderslave1] * pmlyVySlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DyVyPreSlave[inderslave1] + y6);
                                pmlzVzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzVzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzVzPreSlave[inderslave1] + y7);
                                DxVxPreSlave[inderslave1] = y5;
                                DyVyPreSlave[inderslave1] = y6;
                                DzVzPreSlave[inderslave1] = y7;
                                y5 = y5 + pmlxVxSlave[inderslave1];
                                y6 = y6 + pmlyVySlave[inderslave1];
                                y7 = y7 + pmlzVzSlave[inderslave1];

                                pmlzVxSlave[inderslave1] = edzk2Slave[inderslave1] * pmlzVxSlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzVxPreSlave[inderslave1] + y10);
                                pmlxVzSlave[inderslave1] = edxi2Slave[inderslave1] * pmlxVzSlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxVzPreSlave[inderslave1] + y11);
                                DzVxPreSlave[inderslave1] = y10;
                                DxVzPreSlave[inderslave1] = y11;
                                y10 = y10 + pmlzVxSlave[inderslave1];
                                y11 = y11 + pmlxVzSlave[inderslave1];

                                pmlyVxSlave[inderslave1] = edyj2Slave[inderslave1] * pmlyVxSlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DyVxPreSlave[inderslave1] + y8);
                                pmlxVySlave[inderslave1] = edxi2Slave[inderslave1] * pmlxVySlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxVyPreSlave[inderslave1] + y9);
                                DyVxPreSlave[inderslave1] = y8;
                                DxVyPreSlave[inderslave1] = y9;
                                y8 = y8 + pmlyVxSlave[inderslave1];
                                y9 = y9 + pmlxVySlave[inderslave1];

                                pmlzVySlave[inderslave1] = edzk2Slave[inderslave1] * pmlzVySlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzVyPreSlave[inderslave1] + y12);
                                pmlyVzSlave[inderslave1] = edyj2Slave[inderslave1] * pmlyVzSlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DyVzPreSlave[inderslave1] + y13);
                                DzVyPreSlave[inderslave1] = y12;
                                DyVzPreSlave[inderslave1] = y13;
                                y12 = y12 + pmlzVySlave[inderslave1];
                                y13 = y13 + pmlyVzSlave[inderslave1];

                                txxSlaveresult[inderslave1] = txxSlave[inderslave1] + (lamda2u_extSlave[inderslave1] * DT) * y5 + (lamda_extSlave[inderslave1] * DT) * y6 + (lamda_extSlave[inderslave1] * DT) * y7;
                                tyySlaveresult[inderslave1] = tyySlave[inderslave1] + (lamda_extSlave[inderslave1] * DT) * y5 + (lamda2u_extSlave[inderslave1] * DT) * y6 + (lamda_extSlave[inderslave1] * DT) * y7;
                                tzzSlaveresult[inderslave1] = tzzSlave[inderslave1] + (lamda_extSlave[inderslave1] * DT) * y5 + (lamda_extSlave[inderslave1] * DT) * y6 + (lamda2u_extSlave[inderslave1] * DT) * y7;

                                txzSlaveresult[inderslave1] = txzSlave[inderslave1] + (muxzSlave[inderslave1] * DT) * (y10 + y11);
                                txySlaveresult[inderslave1] = txySlave[inderslave1] + (muxySlave[inderslave1] * DT) * (y8 + y9);
                                tyzSlaveresult[inderslave1] = tyzSlave[inderslave1] + (muyzSlave[inderslave1] * DT) * (y12 + y13);
                            }
                        }
                    }
                    else if (_COL == 7 && (StepSlaveX == limitnxt)) // x边缘处只算8个数
                    {
                        for (iy = 1; iy < SlaveNyt1 - 1; iy++)
                        {
                            for (ix = 1; ix < SlaveNxt1 - 3; ix++)
                            {

                                // 核内索引
                                // 1层
                                int inderslave1 = iy * SlaveNxt1 + ix;

                                // 2层
                                int inderslave2 = iy * SlaveNxt1 + ix;
                                int inderslavex2 = iy * SlaveNxt1 + ix + 1;
                                int inderslavexx2 = iy * SlaveNxt1 + ix - 1;
                                int inderslavey2 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslaveyy2 = (iy - 1) * SlaveNxt1 + ix;
                                int inderslavexy2 = (iy + 1) * SlaveNxt1 + ix + 1;

                                y5 = (DOWNvxSlave[inderslave2] - DOWNvxSlave[inderslavexx2]) / H;
                                y6 = (DOWNvySlave[inderslave2] - DOWNvySlave[inderslaveyy2]) / H;
                                y7 = (UPvzSlave[inderslave2] - DOWNvzSlave[inderslave2]) / H;
                                y8 = (DOWNvxSlave[inderslavey2] - DOWNvxSlave[inderslave2]) / H;
                                y9 = (DOWNvySlave[inderslavex2] - DOWNvySlave[inderslave2]) / H;
                                y10 = (UPvxSlave[inderslave2] - DOWNvxSlave[inderslave2]) / H;
                                y11 = (UPvzSlave[inderslavex2] - UPvzSlave[inderslave2]) / H;
                                y12 = (UPvySlave[inderslave2] - DOWNvySlave[inderslave2]) / H;
                                y13 = (UPvzSlave[inderslavey2] - UPvzSlave[inderslave2]) / H;

                                xFHalfTemp2 = -DT * dxi2Slave[inderslave1] * 0.5;
                                xFIntTemp2 = -DT * dxiSlave[inderslave1] * 0.5;
                                yFHalfTemp2 = -DT * dyj2Slave[inderslave1] * 0.5;
                                yFIntTemp2 = -DT * dyjSlave[inderslave1] * 0.5;
                                zFHalfTemp2 = -DT * dzk2Slave[inderslave1] * 0.5;
                                zFIntTemp2 = -DT * dzkSlave[inderslave1] * 0.5;
                                pmlxVxSlave[inderslave1] = edxiSlave[inderslave1] * pmlxVxSlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxVxPreSlave[inderslave1] + y5);
                                pmlyVySlave[inderslave1] = edyjSlave[inderslave1] * pmlyVySlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DyVyPreSlave[inderslave1] + y6);
                                pmlzVzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzVzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzVzPreSlave[inderslave1] + y7);
                                DxVxPreSlave[inderslave1] = y5;
                                DyVyPreSlave[inderslave1] = y6;
                                DzVzPreSlave[inderslave1] = y7;
                                y5 = y5 + pmlxVxSlave[inderslave1];
                                y6 = y6 + pmlyVySlave[inderslave1];
                                y7 = y7 + pmlzVzSlave[inderslave1];

                                pmlzVxSlave[inderslave1] = edzk2Slave[inderslave1] * pmlzVxSlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzVxPreSlave[inderslave1] + y10);
                                pmlxVzSlave[inderslave1] = edxi2Slave[inderslave1] * pmlxVzSlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxVzPreSlave[inderslave1] + y11);
                                DzVxPreSlave[inderslave1] = y10;
                                DxVzPreSlave[inderslave1] = y11;
                                y10 = y10 + pmlzVxSlave[inderslave1];
                                y11 = y11 + pmlxVzSlave[inderslave1];

                                pmlyVxSlave[inderslave1] = edyj2Slave[inderslave1] * pmlyVxSlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DyVxPreSlave[inderslave1] + y8);
                                pmlxVySlave[inderslave1] = edxi2Slave[inderslave1] * pmlxVySlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxVyPreSlave[inderslave1] + y9);
                                DyVxPreSlave[inderslave1] = y8;
                                DxVyPreSlave[inderslave1] = y9;
                                y8 = y8 + pmlyVxSlave[inderslave1];
                                y9 = y9 + pmlxVySlave[inderslave1];

                                pmlzVySlave[inderslave1] = edzk2Slave[inderslave1] * pmlzVySlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzVyPreSlave[inderslave1] + y12);
                                pmlyVzSlave[inderslave1] = edyj2Slave[inderslave1] * pmlyVzSlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DyVzPreSlave[inderslave1] + y13);
                                DzVyPreSlave[inderslave1] = y12;
                                DyVzPreSlave[inderslave1] = y13;
                                y12 = y12 + pmlzVySlave[inderslave1];
                                y13 = y13 + pmlyVzSlave[inderslave1];

                                txxSlaveresult[inderslave1] = txxSlave[inderslave1] + (lamda2u_extSlave[inderslave1] * DT) * y5 + (lamda_extSlave[inderslave1] * DT) * y6 + (lamda_extSlave[inderslave1] * DT) * y7;
                                tyySlaveresult[inderslave1] = tyySlave[inderslave1] + (lamda_extSlave[inderslave1] * DT) * y5 + (lamda2u_extSlave[inderslave1] * DT) * y6 + (lamda_extSlave[inderslave1] * DT) * y7;
                                tzzSlaveresult[inderslave1] = tzzSlave[inderslave1] + (lamda_extSlave[inderslave1] * DT) * y5 + (lamda_extSlave[inderslave1] * DT) * y6 + (lamda2u_extSlave[inderslave1] * DT) * y7;

                                txzSlaveresult[inderslave1] = txzSlave[inderslave1] + (muxzSlave[inderslave1] * DT) * (y10 + y11);
                                txySlaveresult[inderslave1] = txySlave[inderslave1] + (muxySlave[inderslave1] * DT) * (y8 + y9);
                                tyzSlaveresult[inderslave1] = tyzSlave[inderslave1] + (muyzSlave[inderslave1] * DT) * (y12 + y13);
                            }
                        }
                    }
                    else if (_ROW == 7 && (StepSlaveY == limitnyt)) // y边缘处只算8列数
                    {
                        for (iy = 1; iy < SlaveNyt1 - 3; iy++)
                        {
                            for (ix = 1; ix < SlaveNxt1 - 1; ix++)
                            {

                                // 核内索引
                                // 1层
                                int inderslave1 = iy * SlaveNxt1 + ix;

                                // 2层
                                int inderslave2 = iy * SlaveNxt1 + ix;
                                int inderslavex2 = iy * SlaveNxt1 + ix + 1;
                                int inderslavexx2 = iy * SlaveNxt1 + ix - 1;
                                int inderslavey2 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslaveyy2 = (iy - 1) * SlaveNxt1 + ix;
                                int inderslavexy2 = (iy + 1) * SlaveNxt1 + ix + 1;

                                y5 = (DOWNvxSlave[inderslave2] - DOWNvxSlave[inderslavexx2]) / H;
                                y6 = (DOWNvySlave[inderslave2] - DOWNvySlave[inderslaveyy2]) / H;
                                y7 = (UPvzSlave[inderslave2] - DOWNvzSlave[inderslave2]) / H;
                                y8 = (DOWNvxSlave[inderslavey2] - DOWNvxSlave[inderslave2]) / H;
                                y9 = (DOWNvySlave[inderslavex2] - DOWNvySlave[inderslave2]) / H;
                                y10 = (UPvxSlave[inderslave2] - DOWNvxSlave[inderslave2]) / H;
                                y11 = (UPvzSlave[inderslavex2] - UPvzSlave[inderslave2]) / H;
                                y12 = (UPvySlave[inderslave2] - DOWNvySlave[inderslave2]) / H;
                                y13 = (UPvzSlave[inderslavey2] - UPvzSlave[inderslave2]) / H;

                                xFHalfTemp2 = -DT * dxi2Slave[inderslave1] * 0.5;
                                xFIntTemp2 = -DT * dxiSlave[inderslave1] * 0.5;
                                yFHalfTemp2 = -DT * dyj2Slave[inderslave1] * 0.5;
                                yFIntTemp2 = -DT * dyjSlave[inderslave1] * 0.5;
                                zFHalfTemp2 = -DT * dzk2Slave[inderslave1] * 0.5;
                                zFIntTemp2 = -DT * dzkSlave[inderslave1] * 0.5;
                                pmlxVxSlave[inderslave1] = edxiSlave[inderslave1] * pmlxVxSlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxVxPreSlave[inderslave1] + y5);
                                pmlyVySlave[inderslave1] = edyjSlave[inderslave1] * pmlyVySlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DyVyPreSlave[inderslave1] + y6);
                                pmlzVzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzVzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzVzPreSlave[inderslave1] + y7);
                                DxVxPreSlave[inderslave1] = y5;
                                DyVyPreSlave[inderslave1] = y6;
                                DzVzPreSlave[inderslave1] = y7;
                                y5 = y5 + pmlxVxSlave[inderslave1];
                                y6 = y6 + pmlyVySlave[inderslave1];
                                y7 = y7 + pmlzVzSlave[inderslave1];

                                pmlzVxSlave[inderslave1] = edzk2Slave[inderslave1] * pmlzVxSlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzVxPreSlave[inderslave1] + y10);
                                pmlxVzSlave[inderslave1] = edxi2Slave[inderslave1] * pmlxVzSlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxVzPreSlave[inderslave1] + y11);
                                DzVxPreSlave[inderslave1] = y10;
                                DxVzPreSlave[inderslave1] = y11;
                                y10 = y10 + pmlzVxSlave[inderslave1];
                                y11 = y11 + pmlxVzSlave[inderslave1];

                                pmlyVxSlave[inderslave1] = edyj2Slave[inderslave1] * pmlyVxSlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DyVxPreSlave[inderslave1] + y8);
                                pmlxVySlave[inderslave1] = edxi2Slave[inderslave1] * pmlxVySlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxVyPreSlave[inderslave1] + y9);
                                DyVxPreSlave[inderslave1] = y8;
                                DxVyPreSlave[inderslave1] = y9;
                                y8 = y8 + pmlyVxSlave[inderslave1];
                                y9 = y9 + pmlxVySlave[inderslave1];

                                pmlzVySlave[inderslave1] = edzk2Slave[inderslave1] * pmlzVySlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzVyPreSlave[inderslave1] + y12);
                                pmlyVzSlave[inderslave1] = edyj2Slave[inderslave1] * pmlyVzSlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DyVzPreSlave[inderslave1] + y13);
                                DzVyPreSlave[inderslave1] = y12;
                                DyVzPreSlave[inderslave1] = y13;
                                y12 = y12 + pmlzVySlave[inderslave1];
                                y13 = y13 + pmlyVzSlave[inderslave1];

                                txxSlaveresult[inderslave1] = txxSlave[inderslave1] + (lamda2u_extSlave[inderslave1] * DT) * y5 + (lamda_extSlave[inderslave1] * DT) * y6 + (lamda_extSlave[inderslave1] * DT) * y7;
                                tyySlaveresult[inderslave1] = tyySlave[inderslave1] + (lamda_extSlave[inderslave1] * DT) * y5 + (lamda2u_extSlave[inderslave1] * DT) * y6 + (lamda_extSlave[inderslave1] * DT) * y7;
                                tzzSlaveresult[inderslave1] = tzzSlave[inderslave1] + (lamda_extSlave[inderslave1] * DT) * y5 + (lamda_extSlave[inderslave1] * DT) * y6 + (lamda2u_extSlave[inderslave1] * DT) * y7;

                                txzSlaveresult[inderslave1] = txzSlave[inderslave1] + (muxzSlave[inderslave1] * DT) * (y10 + y11);
                                txySlaveresult[inderslave1] = txySlave[inderslave1] + (muxySlave[inderslave1] * DT) * (y8 + y9);
                                tyzSlaveresult[inderslave1] = tyzSlave[inderslave1] + (muyzSlave[inderslave1] * DT) * (y12 + y13);
                            }
                        }
                    }
                    if (
                        !((_COL == 7) && (_ROW == 7) && (StepSlaveY == limitnyt) && (StepSlaveX == limitnxt)) &&
                        !((_COL == 7) && (StepSlaveX == limitnxt)) &&
                        !((_ROW == 7) && (StepSlaveY == limitnyt)))
                    {
                        for (iy = 1; iy < SlaveNyt1 - 1; iy++)
                        {
                            for (ix = 1; ix < SlaveNxt1 - 1; ix++)
                            {

                                // 核内索引
                                // 1层
                                int inderslave1 = iy * SlaveNxt1 + ix;

                                // 2层
                                int inderslave2 = iy * SlaveNxt1 + ix;
                                int inderslavex2 = iy * SlaveNxt1 + ix + 1;
                                int inderslavexx2 = iy * SlaveNxt1 + ix - 1;
                                int inderslavey2 = (iy + 1) * SlaveNxt1 + ix;
                                int inderslaveyy2 = (iy - 1) * SlaveNxt1 + ix;
                                int inderslavexy2 = (iy + 1) * SlaveNxt1 + ix + 1;

                                y5 = (DOWNvxSlave[inderslave2] - DOWNvxSlave[inderslavexx2]) / H;
                                y6 = (DOWNvySlave[inderslave2] - DOWNvySlave[inderslaveyy2]) / H;
                                y7 = (UPvzSlave[inderslave2] - DOWNvzSlave[inderslave2]) / H;
                                y8 = (DOWNvxSlave[inderslavey2] - DOWNvxSlave[inderslave2]) / H;
                                y9 = (DOWNvySlave[inderslavex2] - DOWNvySlave[inderslave2]) / H;
                                y10 = (UPvxSlave[inderslave2] - DOWNvxSlave[inderslave2]) / H;
                                y11 = (UPvzSlave[inderslavex2] - UPvzSlave[inderslave2]) / H;
                                y12 = (UPvySlave[inderslave2] - DOWNvySlave[inderslave2]) / H;
                                y13 = (UPvzSlave[inderslavey2] - UPvzSlave[inderslave2]) / H;

                                xFHalfTemp2 = -DT * dxi2Slave[inderslave1] * 0.5;
                                xFIntTemp2 = -DT * dxiSlave[inderslave1] * 0.5;
                                yFHalfTemp2 = -DT * dyj2Slave[inderslave1] * 0.5;
                                yFIntTemp2 = -DT * dyjSlave[inderslave1] * 0.5;
                                zFHalfTemp2 = -DT * dzk2Slave[inderslave1] * 0.5;
                                zFIntTemp2 = -DT * dzkSlave[inderslave1] * 0.5;
                                pmlxVxSlave[inderslave1] = edxiSlave[inderslave1] * pmlxVxSlave[inderslave1] + xFIntTemp2 * (edxiSlave[inderslave1] * DxVxPreSlave[inderslave1] + y5);
                                pmlyVySlave[inderslave1] = edyjSlave[inderslave1] * pmlyVySlave[inderslave1] + yFIntTemp2 * (edyjSlave[inderslave1] * DyVyPreSlave[inderslave1] + y6);
                                pmlzVzSlave[inderslave1] = edzkSlave[inderslave1] * pmlzVzSlave[inderslave1] + zFIntTemp2 * (edzkSlave[inderslave1] * DzVzPreSlave[inderslave1] + y7);
                                DxVxPreSlave[inderslave1] = y5;
                                DyVyPreSlave[inderslave1] = y6;
                                DzVzPreSlave[inderslave1] = y7;
                                y5 = y5 + pmlxVxSlave[inderslave1];
                                y6 = y6 + pmlyVySlave[inderslave1];
                                y7 = y7 + pmlzVzSlave[inderslave1];

                                pmlzVxSlave[inderslave1] = edzk2Slave[inderslave1] * pmlzVxSlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzVxPreSlave[inderslave1] + y10);
                                pmlxVzSlave[inderslave1] = edxi2Slave[inderslave1] * pmlxVzSlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxVzPreSlave[inderslave1] + y11);
                                DzVxPreSlave[inderslave1] = y10;
                                DxVzPreSlave[inderslave1] = y11;
                                y10 = y10 + pmlzVxSlave[inderslave1];
                                y11 = y11 + pmlxVzSlave[inderslave1];

                                pmlyVxSlave[inderslave1] = edyj2Slave[inderslave1] * pmlyVxSlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DyVxPreSlave[inderslave1] + y8);
                                pmlxVySlave[inderslave1] = edxi2Slave[inderslave1] * pmlxVySlave[inderslave1] + xFHalfTemp2 * (edxi2Slave[inderslave1] * DxVyPreSlave[inderslave1] + y9);
                                DyVxPreSlave[inderslave1] = y8;
                                DxVyPreSlave[inderslave1] = y9;
                                y8 = y8 + pmlyVxSlave[inderslave1];
                                y9 = y9 + pmlxVySlave[inderslave1];

                                pmlzVySlave[inderslave1] = edzk2Slave[inderslave1] * pmlzVySlave[inderslave1] + zFHalfTemp2 * (edzk2Slave[inderslave1] * DzVyPreSlave[inderslave1] + y12);
                                pmlyVzSlave[inderslave1] = edyj2Slave[inderslave1] * pmlyVzSlave[inderslave1] + yFHalfTemp2 * (edyj2Slave[inderslave1] * DyVzPreSlave[inderslave1] + y13);
                                DzVyPreSlave[inderslave1] = y12;
                                DyVzPreSlave[inderslave1] = y13;
                                y12 = y12 + pmlzVySlave[inderslave1];
                                y13 = y13 + pmlyVzSlave[inderslave1];

                                txxSlaveresult[inderslave1] = txxSlave[inderslave1] + (lamda2u_extSlave[inderslave1] * DT) * y5 + (lamda_extSlave[inderslave1] * DT) * y6 + (lamda_extSlave[inderslave1] * DT) * y7;
                                tyySlaveresult[inderslave1] = tyySlave[inderslave1] + (lamda_extSlave[inderslave1] * DT) * y5 + (lamda2u_extSlave[inderslave1] * DT) * y6 + (lamda_extSlave[inderslave1] * DT) * y7;
                                tzzSlaveresult[inderslave1] = tzzSlave[inderslave1] + (lamda_extSlave[inderslave1] * DT) * y5 + (lamda_extSlave[inderslave1] * DT) * y6 + (lamda2u_extSlave[inderslave1] * DT) * y7;

                                txzSlaveresult[inderslave1] = txzSlave[inderslave1] + (muxzSlave[inderslave1] * DT) * (y10 + y11);
                                txySlaveresult[inderslave1] = txySlave[inderslave1] + (muxySlave[inderslave1] * DT) * (y8 + y9);
                                tyzSlaveresult[inderslave1] = tyzSlave[inderslave1] + (muyzSlave[inderslave1] * DT) * (y12 + y13);
                            }
                        }
                    }
                }
                //------------------------------------------回传-----------------------------------------------------------
                // 向主核中传数组DMA传输
                if (StepSlaveZ > 0)
                {
                    for (int conveyy = 1; conveyy < (SlaveNyt1 - 1); conveyy++)
                    {
                        if ((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3)))
                        {
                        }
                        if (!((_ROW == 7) && (StepSlaveY == limitnyt) && (conveyy > (SlaveNyt1 - 3))))
                        {
                            if (_COL == 7 && (StepSlaveX == limitnxt))
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                // 回传所以需要从每个从核中的第二列开始算也就是x方向+1的位置开始取10个数
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt) + 1;
                                // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt1 + 1;
                                // 9A
                                // athread_dma_put(&txx[transferinder], &txxSlaveresult[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&tyy[transferinder], &tyySlaveresult[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&tzz[transferinder], &tzzSlaveresult[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&txy[transferinder], &txySlaveresult[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&txz[transferinder], &txzSlaveresult[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&tyz[transferinder], &tyzSlaveresult[getindex], sizeof(float) * (SlaveNxt - 2));
                                // // athread_dma_put(&dxii[transferinder], &dxiSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // // athread_dma_put(&mu_extaccept[transferinder], &mu_extSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlxVx[transferinder], &pmlxVxSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlyVy[transferinder], &pmlyVySlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlzVz[transferinder], &pmlzVzSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlzVx[transferinder], &pmlzVxSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlxVz[transferinder], &pmlxVzSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlyVx[transferinder], &pmlyVxSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlxVy[transferinder], &pmlxVySlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlzVy[transferinder], &pmlzVySlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&pmlyVz[transferinder], &pmlyVzSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DxVxPre[transferinder], &DxVxPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DyVyPre[transferinder], &DyVyPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DzVzPre[transferinder], &DzVzPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DzVxPre[transferinder], &DzVxPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DxVzPre[transferinder], &DxVzPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DyVxPre[transferinder], &DyVxPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DxVyPre[transferinder], &DxVyPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DzVyPre[transferinder], &DzVyPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));
                                // athread_dma_put(&DyVzPre[transferinder], &DyVzPreSlave[getindex], sizeof(float) * (SlaveNxt - 2));

                                // 5A
                                put_reply = 0;
                                athread_put(PE_MODE, &txxSlaveresult[getindex], &txx[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &tyySlaveresult[getindex], &tyy[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &tzzSlaveresult[getindex], &tzz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &txySlaveresult[getindex], &txy[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &txzSlaveresult[getindex], &txz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &tyzSlaveresult[getindex], &tyz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxVxSlave[getindex], &pmlxVx[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlyVySlave[getindex], &pmlyVy[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzVzSlave[getindex], &pmlzVz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzVxSlave[getindex], &pmlzVx[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxVzSlave[getindex], &pmlxVz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlyVxSlave[getindex], &pmlyVx[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxVySlave[getindex], &pmlxVy[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzVySlave[getindex], &pmlzVy[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlyVzSlave[getindex], &pmlyVz[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxVxPreSlave[getindex], &DxVxPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DyVyPreSlave[getindex], &DyVyPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzVzPreSlave[getindex], &DzVzPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzVxPreSlave[getindex], &DzVxPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxVzPreSlave[getindex], &DxVzPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DyVxPreSlave[getindex], &DyVxPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxVyPreSlave[getindex], &DxVyPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzVyPreSlave[getindex], &DzVyPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                athread_put(PE_MODE, &DyVzPreSlave[getindex], &DyVzPre[transferinder], sizeof(float) * (SlaveNxt - 2), &put_reply, 0, 0);
                                while (put_reply != 24)
                                    ;
                            }
                            if (!(_COL == 7 && (StepSlaveX == limitnxt)))
                            {
                                // 传输数据DMA对应在模型上的坐标索引
                                // (nxt * nyt * (StepSlaveZ + conveyz)) 到对应的那一层的首地址  在z方向上
                                // (nxt * StepSlaveY) 到索引所在从核组的那一列的首地址
                                // (StepSlaveX) 到索引所在的核组的首地址
                                // (_ROW * SlaveNyt * nxt) 到索引所在的列的从核首地址
                                // (_COL * SlaveNxt) 将对应的从核的行号进行对应
                                // (conveyy * nxt) 到索引所在的列了
                                // 回传所以需要从每个从核中的第二列开始算也就是x方向+1的位置开始取10个数
                                int transferinder = (nxt * nyt * (StepSlaveZ)) + (nxt * StepSlaveY) + (StepSlaveX) + (_ROW * SlaveNyt * nxt) + (_COL * SlaveNxt) + (conveyy * nxt) + 1;
                                // 每个从核中(conveyz * SlaveNxt * SlaveNyt) + conveyy * SlaveNxt  所有从核是并行取数据的 所以从核中的索引是一样的
                                int getindex = conveyy * SlaveNxt1 + 1;
                                // 9A
                                // athread_dma_put(&txx[transferinder], &txxSlaveresult[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&tyy[transferinder], &tyySlaveresult[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&tzz[transferinder], &tzzSlaveresult[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&txy[transferinder], &txySlaveresult[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&txz[transferinder], &txzSlaveresult[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&tyz[transferinder], &tyzSlaveresult[getindex], sizeof(float) * SlaveNxt);
                                // // athread_dma_put(&dxii[transferinder], &dxiSlave[getindex], sizeof(float) * SlaveNxt);
                                // // athread_dma_put(&mu_extaccept[transferinder], &mu_extSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlxVx[transferinder], &pmlxVxSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlyVy[transferinder], &pmlyVySlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlzVz[transferinder], &pmlzVzSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlzVx[transferinder], &pmlzVxSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlxVz[transferinder], &pmlxVzSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlyVx[transferinder], &pmlyVxSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlxVy[transferinder], &pmlxVySlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlzVy[transferinder], &pmlzVySlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&pmlyVz[transferinder], &pmlyVzSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DxVxPre[transferinder], &DxVxPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DyVyPre[transferinder], &DyVyPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DzVzPre[transferinder], &DzVzPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DzVxPre[transferinder], &DzVxPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DxVzPre[transferinder], &DxVzPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DyVxPre[transferinder], &DyVxPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DxVyPre[transferinder], &DxVyPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DzVyPre[transferinder], &DzVyPreSlave[getindex], sizeof(float) * SlaveNxt);
                                // athread_dma_put(&DyVzPre[transferinder], &DyVzPreSlave[getindex], sizeof(float) * SlaveNxt);

                                // 5A
                                put_reply = 0;
                                athread_put(PE_MODE, &txxSlaveresult[getindex], &txx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &tyySlaveresult[getindex], &tyy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &tzzSlaveresult[getindex], &tzz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &txySlaveresult[getindex], &txy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &txzSlaveresult[getindex], &txz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &tyzSlaveresult[getindex], &tyz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxVxSlave[getindex], &pmlxVx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlyVySlave[getindex], &pmlyVy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzVzSlave[getindex], &pmlzVz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzVxSlave[getindex], &pmlzVx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxVzSlave[getindex], &pmlxVz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlyVxSlave[getindex], &pmlyVx[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlxVySlave[getindex], &pmlxVy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlzVySlave[getindex], &pmlzVy[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &pmlyVzSlave[getindex], &pmlyVz[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxVxPreSlave[getindex], &DxVxPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DyVyPreSlave[getindex], &DyVyPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzVzPreSlave[getindex], &DzVzPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzVxPreSlave[getindex], &DzVxPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxVzPreSlave[getindex], &DxVzPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DyVxPreSlave[getindex], &DyVxPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DxVyPreSlave[getindex], &DxVyPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DzVyPreSlave[getindex], &DzVyPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                athread_put(PE_MODE, &DyVzPreSlave[getindex], &DyVzPre[transferinder], sizeof(float) * SlaveNxt, &put_reply, 0, 0);
                                while (put_reply != 24)
                                    ;
                            }
                        }
                    }
                }
                //-------------------------------------------------从核内重新赋值---------------------------------------------------------
                // 当stepSlaveZ == 0
                /* if (StepSlaveZ == 0)
                {
                    // 每次取上层的值 放到下层

                    MIDDLEvxSlave = DOWNvxSlave;
                    DOWNvxSlave = UPvxSlave;
                    UPvxSlave = MIDDLEvxSlave;

                    MIDDLEvySlave = DOWNvySlave;
                    DOWNvySlave = UPvySlave;
                    UPvySlave = MIDDLEvySlave;
                }
                // 二层数值的参数循环幅值 将最上层值传到中间，中间值传到底面
                if (StepSlaveZ > 0)
                {
                    // 每次取上层的值 放到下层

                    MIDDLEvxSlave = DOWNvxSlave;
                    DOWNvxSlave = UPvxSlave;
                    UPvxSlave = MIDDLEvxSlave;

                    MIDDLEvySlave = DOWNvySlave;
                    DOWNvySlave = UPvySlave;
                    UPvySlave = MIDDLEvySlave;

                    MIDDLEvzSlave = DOWNvzSlave;
                    DOWNvzSlave = UPvzSlave;
                    UPvzSlave = MIDDLEvzSlave;
                } */
            }
        }
    }
}
