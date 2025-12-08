# 使用方法
## 项目设置

操作方法：编辑 `settings.py`

该脚本为全局配置。包含项目路径管理（PROJECT_ROOT）、数据集配置（DATASETS / ACTIVE_DATA）、全局性的数据清理、切窗与特征提取参数（PARAMS）、信号别名规范（ SIGNAL_ALIASES / DEVICE_TAGS）、数据定义规范（ SCHEMA ）、事件标注规范(EVENTS_CANON),以及输出数据长表中包含的信息（FINAL_LABELS）

## 载入数据

操作方法：运行 `1_load_data.py`

数据集信息在 settings 的 DATASETS 中设置。ACTIVE_DATA 指定使用的数据集。`local` 为使用者自己录制的数据默认名称，还可添加其他测试数据。

settings.DATASETS 中与载入数据相关的 key:
- `loader`：每个数据集都需要专门写一个数据调用代码，负责该数据集的调用，代码放在`scripts\loaders\`下，`loader`的值为该脚本的路径。`1_load_data.py`会根据此路径调用相应的脚本执行数据载入工作
- `paths`：不同数据处理阶段数据保存路径

local 数据为 LSL 流记录的数据，需要提前转换为 .csv 文件，文件名类似于 `sub-P002_ses-S001_task-Default_run-001_all_rr_h10`，为被试号+测试号+任务标签+试次号的关键词组合。将所有数据放在同一个文件夹中，运行 `1_load_data.py`，程序会打开系统文件夹，要求用户指定数据所在文件夹。指定文件夹后，程序将数据载入本项目目录，目录可在 `paths` 中设置。载入后，文件名会做一些简化处理，如 `P002S001T001R001_rr`。

载入数据默认保存在 `data/raw/ACTIVE_DATA`路径下。

注：事件数据名会从 markers 重命名为 events

## 数据规范化

操作方法：`2_data_norm.py`

该操作的目的是统一信号及属性的定义名称。该规范定义在settings.SIGNAL_ALIASES、settings.SCHEMA 中。未来数据保存时对不同的信号属性进行命名时均遵循此规范。数据规范包含三个部分：对数据命名的规范，包括被试号与信号类型、统一信号的属性名、基于规范筛查数据。

规范化的数据默认保存在 `data/processed/norm/ACTIVE_DATA`路径下。acc, ecg, resp等定义为连续数据，保存为parquet文件，在该目录下的preview文件夹提供对parquet的csv预览文件。hr, rr, events保存为csv文件。

## 确定RR数据

操作方法：`3_select_rr.py`。需要运行两次，第一次使用者需要确定使用哪个RR数据，第二次生成最终“确认RR数据”。

该脚本的目的是从现有数据中获取质量最佳的RR数据。其中“选择”在于最终确认的RR是来自设备提供的RR数据，还是通过ECG转换后的RR数据。运行时，脚本会在 SRC_NORM_DIR 中扫 *_rr.csv 与 *_ecg.*文件。如果存在 *_rr.csv，就进入“双源比较”流程，即在两个数据源中选择最佳RR，否则直接从ECG生成RR。

选择的思路是首先计算两路 RR 的重叠区间，如果重叠区间太短（小于10秒）则警告退出。否则将两路RR映射为 1 Hz 心率轨用于对比与绘图。绘图的范围基于顶部全局变量 PLOT_RANGE_MODE，'auto' 模式下，如果存在 events，优先使用 events 的最小到最大时间作为横轴范围。

第一步会在 `data/processed/rr_select/` 路径下生成 decision_suggested.csv 数据对比表格，其中给出多项评估哪个RR更好的指标。在`choice_suggested.csv`列下给出建议确认的RR (ecg_rr / device_rr),使用者可手动修改此列，同时 **将该表从 decision_suggested.csv 修改为 decision.csv 确认最终选择决定**。

第二步是生成确认数据,如果 `decision.csv` 已经存在，则会根据该决定表中各个被试的 `choice_suggested`列中的值生成最终确定的RR数据。数据保存在`/data/processed/confirmed/` 路径，同时在 `/data/processed/rr_select/ ACTIVE_DATA/` 路径生成 final_rr_summary.csv 报告生成文件的名字和选择rr的策略。

此步骤还会生成RR数据预览图。根据 settings.DATASET[ACTIVE_DATA]["preview_sids"] 列表中的用户ID绘制。如果列表为空，则只绘制第一个被试的结果。

该脚本有个重要的参数 ECG_RR_V = 'v2' 有v1 v2 两个选项，v1是用 NeuroKit2 做的，但数据会失真。v2 算法保守，但会有很多地方识别错误，需要在 clean_rr 中进行专门的清理。

## 清理 RR 

操作方法：`4a_clean_rr.py`

该脚本用于处理rr数据中的异常尖峰。操作的数据对象为 confirmed rr。执行该脚本前必须先执行 `3_select_rr.py`，并且在rr_select/路径下有decision.csv文件。`4b_clean_rr.py`会根据该文件中标注为来自 device_rr 的数据进行分析。分析后会给出一份汇总报告 `clean_rr_summary.csv`, 

脚本有四种清理方法，分别为
- "interp": 使用neurokit2 的插值方法处理
- "split" : 针对那种漏检的情况，例如 700, 1400, 700 在这个情况下，一般1400是这个点是漏检了一个700，所以要先把这个值拆成两个700。
- "delete": 已识别的短段直接删除，不做插值
- "fill": 删掉一个异常点，用前后两个点的平均值替换

这些方法需要在全局变量 RR_REPAIR_METHOD  = [] 来设定。注意，如果数组中包含了 split 方法，一定会先处理此类数据。

标注对哪些被试的 RR 数据进行处理。n_segments 表示在该被试 RR 中识别到的“短时尖峰短段”的数量。n_corrected 表示需处理的搏点个数之和。ratio_corrected 表示n_corrected / 全部 RR 点数 的比例。warning 当数据糟糕到一定程度会提醒，例如异常短段数达到提醒阈值（默认 4）或需修正搏点占比达到提醒阈值（默认 0.05）

处理后的 RR 会覆盖之前的confirmed rr。

## 绘图查看

操作方法：`4b_preview.py`

该脚本绘制上一步“确定选择RR”的预览图。用于快速检查RR是否存在问题。预览图根据 settings.DATASET[ACTIVE_DATA]["preview_sids"] 列表中的用户ID绘制。根据 settings.DATASET[ACTIVE_DATA]["windowing"]["apply_to"] 列表中的信号类型绘制信号数据，前提是对应的数据必须在 confirmed/文件夹下有一份拷贝。events, ecg的数据放在 norm/ 下。
对视图的控制可以通过几个全局变量
- RR_RAW_FORCE_YLIM = (300, 1200) 设置RR图的 Y 轴大小，默认300-1200 ms是个合理区间
- HR1HZ_FORCE_YLIM = (40, 120) 设置hr图的 Y 轴大小，常人心跳一般在40-120之间
- BR_FORCE_YLIM = (6, 30) 呼吸值的 Y 轴区间
- RESP_MAX_BPM = 30 当记录时间较长，呼吸图会比较密。此值控制曲线的最小间距，控制稀疏程度。越大越稀疏
- ECG_PLOT_SPAN = 40.0 绘制指定窗口大小的区域的ecg，以方便放大某一时间段的ecg 做深入检查。⚠️ 起始时间需在 main 中设置，可使用 event 标签
- PRE_SID = [..] 输入想要预览的 sid，程序会根据这些 sid 生成结果
此外，该方法可以放大一段ECG数据。修改 make_windowing_ecg_plot 的参数，设置开始点和时长：
`
make_windowing_ecg_plot(
    sid=sid, 
    event="Baseline",  # <--- 设置1：起始点（通过事件名查找）
    t_start=None,      # <--- 设置2：起始点（手动指定秒数，如果 event 为 None）
    span=10.0          # <--- 设置3：时长（秒）
)
`

## 切窗

操作方法：`5a_windowing.py`

首先需要在settings中设置 `windowing` 切窗设置。首要的两个设置为`use`和`method`。

`method`设置切窗方法，包括全切与细分2种（cover / subdivide）。全切不考虑切窗历史，细分则考虑上一次切窗历史，对上一次切窗结果中的某一部分进行再次切窗。第一次cover的结果会放在 processed/windowing/local/level1下。对上一次切窗结果的每次“细分”切窗会自动增加一极level。每次切窗会生成一个log文件供使用者追溯各切窗之间的关系，以及一个index文件便于后续对所有的窗口的整合组装。

选择 subdivide 模式会需要在终端进行一些交互处理。要求手动选中“某一层的某一个窗口文件”，代码据此识别是哪个窗口（w03 / w04…），然后在这一层的 index.csv 里找到所有被试的同一个父窗口区间，对所有被试在同一父窗内统一做“二次切窗”。

`use`设置切窗策略，可选策略包括根据事件信息切窗、切单窗、滑窗、事件+单窗、事件+滑窗、单窗+滑窗，(events / single / sliding / events_offset / events_sliding / single_sliding) 共6种。使用 events 切窗需要事件表，在 `events_path`下设置路径。开始时间为相对位置，例如start_s 设置为12，会从数据12秒的位置作为切窗开始位置。

推荐使用 `events_labeled_windows` 方法。该方法根据顺序与事件名来精确定义窗口。用户根据 events 列表中的事件顺序，在settings中逐一写下每窗的起始事件标签。

## 整理切窗数据

操作方法：`5b_collect_windowing.py`

将所有的切窗整理为后续可用的数据。数据保存在windowing/collected/下。数据按照信号类型放在不同的文件夹。同时还生成一个collected_index.csv，供后续整理心率指标的长表使用。

## 切窗数据绘图

操作方法：`5c_window_plot.py`

读取 windowing/collected 文件夹的切窗数据，为制定窗口和制定被试绘图。
- SID = ["..."] 输入想要查看的被试ID
- WIN = ["..."] 输入想要查看的窗口ID

## 计算每窗的指标

操作方法：`6_gather_features.py`

该脚本调用scripts/features/下的计算时域、频域和 RSA 的API 对各窗数据计算相应的指标。形成一个长表。

## 生成实验组编号

操作方法：`7_group_by_sid.py`

根据数据名来推断实验组。在 local 数据中，文件名的前缀为 P002S001T001R001 的格式。在settings 中标注 "groups" 的识别方式，指明哪个字符到哪个字符为实验组的信息。例如 "task":    {"start": 9, "end": 12} 则表示 T 后面的 001 为实验组信息。在 "value" 中标注不同字段所表示的组号。

生成的信息保存在 data/groups中

## 合成生理指标长表

操作方法：`8_make_physico_table.py`

该脚本将 `7_group_by_sid.py`生成的分组号信息加入到 `6_gather_features.py` 生成的数据里。生成最终的生理数据 physico 表，该表可用于spss 的混合线性模型分析。

生成的信息保存在 data/processed/physico/physico.csv

## 构建心理指标宽表

操作方法：`9_collect_psycho.py`

该脚本构建心理指标宽表。在使用该脚本前，需要分别将各时间段的心理测量结果保存为 t0.csv, t1.csv, t2.csv 等格式，设定好心理指标名称，时间性测量指标要加入 `t0_` 的前缀，比如基线、诱导和干预三个阶段的状态焦虑指标为 t0_stai, t1_stai, t2_stai。非时间性指标不要加前缀，直接写flow, peou等指标名称。将所有 csv 放在同一个文件夹下。
所有文件要预先计算出几个指标的的值（计算平均值，参考提供的snytax），标注好被测id（subject_id列，与生理数据保持一致）。

在 settings 中 "psycho_indices" 列出程序需要调用的指标。程序运行后会跳出系统窗口，可交互式地指定csv所在路径。生成心理指标表

生成的信息保存在 data/psycho/psycho.csv

## 建立宽表

操作方法：`10_make_wide_table.py`

该脚本把 physico 和 psycho 两个表拼合，并按照时间维度展开，形成宽表。该表主要用于 spss 重复方差分析。TABLE_TYPE = "psy" 可选 psy, phy, both 参数可用于制定生成生理、心理还是全部数据宽表。主要目的用于单独分析

生成的信息保存在 data/final/wide_table.csv

# 复杂切窗数据的宽表与长表

操作方法：`11_make_ELW_table.py`

该脚本针对包含测试中与测试后的复杂切窗模型设计的。专门用于对使用 settings 中 events_labeled_windows 切窗模式生成的数据进行制表。使用此脚本制表需要在 events_labeled_windows 中 设置窗口的命名规范。由于有测试期（phase）和测试间（gap）之分，测试期表示被试在静息、诱导、干预的整个时间段；测试间表示每个测试期结束，以及下一个测试期开始前，用户的回答问卷以及静止的期间。这两个期间在分析时是按照两套时间维度对待的，因此对制表有特殊的要求。在使用此方法制表时需要为每个窗口名（windows["name"]）有个明确的标注。改标注放在"window_category"的列表里。例如 "p" 表示 phase,那么测试期的窗口名要加上 p_的前缀，如 p_baseline；"g_" 表示 gap, 测试间的窗口名则使用类似 "g_t1" 的名字类型。"psycho_time" 键值表示心理指标测量是在哪个时间类别测量的，比如 "psycho_time" = "g" 表示心理指标是在测试间测量的。程序会根据这些规则来识别不同指标依赖哪些时间，并进行制表。该脚本会制作宽表与长表

宽表数据保存在 data/final/wide_table.csv；长表数据保存在 data/final/long_table.csv

# 对数据结果进行基本数据描述
分析内容包括
- 每个被试的均值 / 标准差 / CV
    - desc_rmssd_ms_by_subject_id.csv
    - 每一行是一个被试一整个实验期内的总体描述，用来看“这个人整体高还是整体低、稳定不稳定”

- 每个被试 × 时间窗口的描述统计
    - desc_hf_log_ms2_by_subject_id_and_t_id.csv
   	- 直接查看某个被试在 T1、T2、T3 各自的均值 / 标准差 / CV。
	- 很适合在 Excel 里按行画图，看个体的时间趋势是否和组均值方向一致，谁是“反向运动员”

- 谁在放大整体方差：全局层面
    - variance_influence_overall_hf_log_ms2_by_subject_id.csv
    - n：该被试数据点个数
	- mean：该被试的平均水平
	- sd：该被试的标准差
	- mean_diff_from_overall：被试均值减去总体均值
	- between_ss_contrib：此被试在 Between SS 中的那一项 n_i (\bar{y}_i - \bar{y})^2
	- between_ss_contrib_pct_of_between：该被试贡献 / 被试间平方和
	- between_ss_contrib_pct_of_total：该被试贡献 / 总平方和
	- mean_z_across_subjects：以“被试均值”在被试群体内计算 Z 分数，帮助识别极端高低值被试
    - 文件已经按 between_ss_contrib_pct_of_total 从大到小排序，反映了“对总体方差贡献最大的几个被试”

- 分条件组内：谁在拖累本组的方差
    - variance_influence_by_task_hf_log_ms2_by_subject_id.csv
    - 在 “点击填色” 组里，是哪几个被试拉高了组内均值差异？
	- 在 “笔触涂抹” 组里，是不是另一些人特别极端？

- 分条件组内的被试轨迹图
    - traj_rmssd_ms_by_subject_id_and_t_id_cond_click.png
    - 在实验组内部，是不是大部分被试都按“理论方向”变化（例如 T2 上升，T3 下降），比较控制组是不是完全躺平或反向

上述这些分析结果可用于对数据进行基本评估。包括：

- 筛出“极端被试”做敏感性分析
	- 在 variance_influence_overall_* 里，把 between_ss_contrib_pct_of_total 排序，挑出贡献特别高的 1–2 个被试。
	- 可以做两套分析：
	- 全部被试
	- 排除这些极端被试
	- 比较组间差异、交互项是否稳定。
- 对照个体趋势与组均值趋势是否一致
	- 对 desc_{var}_by_subject_id_and_t_id.csv 中某个变量查看
	- 对 traj_{var}_by_subject_id_and_t_id_cond_{cond_sanitized}.png 查看
	- 哪些被试是 T1→T2→T3 按“理论方向”变化；
	- 哪些人走反了
- 用 CV 看数据质量
	- cv 很大而 mean 很小的被试，可能存在噪声特别大的窗或清理不彻底。
	- 可以把 CV 排序选前若干名，回去看这些被试对应的 RR / ECG 图。

# Z score 分析数据

操作方法：`12_cal_Zscore.py`

该脚本生成用于 Z score 分析的数据。该数据表完全为 spss 的后续分析准备。脚本执行后会打印 变量与值的标签，可直接粘贴到spss syntax 中运行。

生成的信息保存在 data/final/zscore_table.csv