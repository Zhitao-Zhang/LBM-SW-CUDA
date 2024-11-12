# LBM-SW-CUDA
LBM的优化


*****************************************************************************************************************************
                                                      CUDA
*****************************************************************************************************************************

System requirement
To ensure that the program runs smoothly, please meet the following system configuration requirements:

Graphics card: NVIDIA RTX 3090 or higher performance graphics card
CUDA version: 11.5 or higher

Instructions

Run the program directly, and the parameters are modified in the LOPOM.cu file.

*****************************************************************************************************************************
                                                      Sunway
*****************************************************************************************************************************

Step

**********************************************************************
                           基础
**********************************************************************
#!/bin/bash
mpicc -mhost -mieee -c main.c
sw5gcc -mslave -mieee -msimd -w -c slave.c
mpicc -mhybrid main.o slave.o -lm_slave -o a.out

sh.run.sh
bsub -I -J zzt -q q_sw_share -n 125 -cgsp 64 -sw3run swrun-5a -b -host_stack 256 -share_size 5000 ./a.out


**********************************************************************
                         SIMD
**********************************************************************
#!/bin/bash
mpicc -mhost -mieee -c main.c
sw5gcc -mslave -mieee -msimd -w -c slaveSIMD.c
mpicc -mhybrid main.o slaveSIMD.o -lm_slave -o a.out

sh.run.sh
bsub -I -J zzt -q q_sw_share -n 125 -cgsp 64 -sw3run swrun-5a -b -host_stack 256 -share_size 5000 ./a.out


Contact information
If you have questions or need support, please contact us: zzt21@mails.jlu.edu.cn

