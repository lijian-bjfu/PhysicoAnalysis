* Encoding: UTF-8.
* 1. 反向计分 (Compute) Q1_Row1, Q1_Row4, Q1_Row5.
COMPUTE Q1_Row1_R = 5 - Q1_Row1.
COMPUTE Q1_Row4_R = 5 - Q1_Row4.
COMPUTE Q1_Row5_R = 5 - Q1_Row5.
EXECUTE.

* 2. 计算均值 (同上).
COMPUTE t3_stai = MEAN(Q1_Row1_R, Q1_Row2, Q1_Row3, Q1_Row4_R, Q1_Row5_R, Q1_Row6).
VARIABLE LABELS  t3_stai 'SATI 状态焦虑量表总均值'.

* --- 计算状态心智正念 (SMM) 与身体正念 (SBM) 量表 ---.
* 1. 计算 "状态心智正念" (smm) 均值 (共 14 项).
COMPUTE t3_smm = MEAN(
    Q2_Row1, Q2_Row3, Q2_Row4, Q2_Row5, Q2_Row6, Q2_Row7, 
    Q2_Row10, Q2_Row11, Q2_Row12, Q2_Row15, Q2_Row16, 
    Q2_Row17, Q2_Row19, Q2_Row20
).
VARIABLE LABELS t3_smm '状态心智正念 (SMM) 均值'.
EXECUTE.

* 2. 计算 "状态身体正念" (sbm) 均值 (共 7 项).
COMPUTE t3_sbm = MEAN(
    Q2_Row2, Q2_Row8, Q2_Row9, Q2_Row13, Q2_Row14, 
    Q2_Row18, Q2_Row21
).
VARIABLE LABELS t3_sbm '状态身体正念 (SBM) 均值'.

* 3. 计算 "总体正念" (sbs) 均值.
COMPUTE t3_sbs = MEAN(
    Q2_Row1, Q2_Row3, Q2_Row4, Q2_Row5, Q2_Row6, Q2_Row7, 
    Q2_Row10, Q2_Row11, Q2_Row12, Q2_Row15, Q2_Row16, 
    Q2_Row17, Q2_Row19, Q2_Row20,
    Q2_Row2, Q2_Row8, Q2_Row9, Q2_Row13, Q2_Row14, 
    Q2_Row18, Q2_Row21
).
VARIABLE LABELS t3_sbs '状态正念 (SBS) 均值'.

* --- 计算 FSS 心流量表 (Q3_Row1 - Q3_Row33) ---.

* 1. 挑战-技能平衡 (Balance).
COMPUTE fss_balance = MEAN(Q3_Row1, Q3_Row8, Q3_Row17, Q3_Row25).

* 2. 明确目标 (Goals).
COMPUTE fss_goal = MEAN(Q3_Row2, Q3_Row10, Q3_Row27).

* 3. 明确反馈 (Feedback).
COMPUTE fss_feedback = MEAN(Q3_Row3, Q3_Row11, Q3_Row19, Q3_Row28).

* 4. 专注于任务 (Concentration).
COMPUTE fss_concentration = MEAN(Q3_Row4, Q3_Row12, Q3_Row20, Q3_Row29).

* 5. 自我意识消失 (Selflessness).
COMPUTE fss_selflessness = MEAN(Q3_Row5, Q3_Row14, Q3_Row22, Q3_Row31).

* 6. 时间感改变 (Time).
COMPUTE fss_time = MEAN(Q3_Row6, Q3_Row15, Q3_Row23, Q3_Row32).

* 7. 自体目的性体验 (Autotelic).
COMPUTE fss_autotelic = MEAN(Q3_Row7, Q3_Row16, Q3_Row24, Q3_Row33).

* 8. 行动与意识融合 (Merging).
COMPUTE fss_merging = MEAN(Q3_Row9, Q3_Row18, Q3_Row26).

* 9. 控制感 (Control).
COMPUTE fss_control = MEAN(Q3_Row13, Q3_Row21, Q3_Row30).

* 添加变量标签 (Variable Labels) 方便查看.
VARIABLE LABELS
    fss_balance 'FSS 1. 挑战-技能平衡'
    fss_goal 'FSS 2. 明确目标'
    fss_feedback 'FSS 3. 明确反馈'
    fss_concentration 'FSS 4. 专注于任务'
    fss_selflessness 'FSS 5. 自我意识消失'
    fss_time 'FSS 6. 时间感改变'
    fss_autotelic 'FSS 7. 自体目的性体验'
    fss_merging 'FSS 8. 行动与意识融合'
    fss_control 'FSS 9. 控制感'.
EXECUTE.

* --- 计算 FSS 心流总指标 (t3_fc) ---.
COMPUTE fss = MEAN(Q3_Row1 TO Q3_Row33).
VARIABLE LABELS fss 'FSS 心流总均值 (33项)'.

* --- 计算 PEOU 易用性量表 (t3_peou) ---.
COMPUTE peou = MEAN(Q4_Row1 TO Q4_Row5).
VARIABLE LABELS peou 'PEOU 易用性量表均值'.

* --- 计算 投入量表 (t3_effort) ---.
COMPUTE effort = MEAN(Q5_Row1, Q5_Row2, Q5_Row3).
VARIABLE LABELS effort '投入量表均值'.

* --- 计算 PANA 正面情绪量表 (t3_pa) ---.
COMPUTE pa = MEAN(Q6_Row1, Q6_Row2, Q6_Row3, Q6_Row4).
VARIABLE LABELS pa 'PANA 正面情绪 (PA) 均值'.

EXECUTE.
