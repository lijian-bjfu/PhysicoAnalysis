* Encoding: UTF-8.
* 1. 反向计分 (Compute) Q1_Row1, Q1_Row4, Q1_Row5.
COMPUTE Q1_Row1_R = 5 - Q1_Row1.
COMPUTE Q1_Row4_R = 5 - Q1_Row4.
COMPUTE Q1_Row5_R = 5 - Q1_Row5.
EXECUTE.

* 2. 计算均值 (同上).
COMPUTE t2_stai = MEAN(Q1_Row1_R, Q1_Row2, Q1_Row3, Q1_Row4_R, Q1_Row5_R, Q1_Row6).
VARIABLE LABELS  t2_stai 'SATI 状态焦虑量表总均值'.

* --- 计算状态心智正念 (SMM) 与身体正念 (SBM) 量表 ---.
* 1. 计算 "状态心智正念" (smm) 均值 (共 14 项).
COMPUTE t2_smm = MEAN(
    Q2_Row1, Q2_Row3, Q2_Row4, Q2_Row5, Q2_Row6, Q2_Row7, 
    Q2_Row10, Q2_Row11, Q2_Row12, Q2_Row15, Q2_Row16, 
    Q2_Row17, Q2_Row19, Q2_Row20
).
VARIABLE LABELS t2_smm '状态心智正念 (SMM) 均值'.
EXECUTE.

* 2. 计算 "状态身体正念" (sbm) 均值 (共 7 项).
COMPUTE t2_sbm = MEAN(
    Q2_Row2, Q2_Row8, Q2_Row9, Q2_Row13, Q2_Row14, 
    Q2_Row18, Q2_Row21
).
VARIABLE LABELS t2_sbm '状态身体正念 (SBM) 均值'.

* 3. 计算 "总体正念" (sbs) 均值.
COMPUTE t2_sbs = MEAN(
    Q2_Row1, Q2_Row3, Q2_Row4, Q2_Row5, Q2_Row6, Q2_Row7, 
    Q2_Row10, Q2_Row11, Q2_Row12, Q2_Row15, Q2_Row16, 
    Q2_Row17, Q2_Row19, Q2_Row20,
    Q2_Row2, Q2_Row8, Q2_Row9, Q2_Row13, Q2_Row14, 
    Q2_Row18, Q2_Row21
).
VARIABLE LABELS t2_sbs '状态正念 (SBS) 均值'.

EXECUTE.
