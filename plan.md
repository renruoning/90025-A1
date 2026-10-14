# COMP90025 Project 2 实现计划

## 背景速览

- 任务:MPI + OpenMP 混合实现分布式 BPE(分词计数 + 每轮 256 个 pair 的合并)。
- 截止时间:10 月 20 日(周二)23:00 AEDT。演示 30 分钟,必须预约:https://go.unimelb.edu.au/vo4h
- 评分(满分 20):正确性 2(4 个配置各 0.5)、速度 8、SLURM 2、报告 8。每项最终得分乘演示系数(0 到 1)。
- 四种配置(单节点,共 16 线程):{4,4,4,4}、{4,4,8}、{2,6,8}、{2,4,10}。
- 速度分:S = 4·T_baseline / (T1+T2+T3+T4),S>3.2 满分。前提是四个配置输出都与 `bin/bpe_seq` 逐字节一致。

## 关键规则(出错会清零或扣分)

- 只有 rank 0 读输入、写 output.txt(R2.6)。
- 不能只有一个 rank 干活,其他 rank 只转发(R2.3)。
- 不能写死线程数、不能针对配置特判(R1.4)。
- 必须调用 MPI(R2.2)。
- 频繁 git 提交,作者邮箱用学生邮箱(R2.4)。
- 在 Spartan 上能编译、无运行时错误(R2.5)。
- tar.gz 不含二进制、输入、输出文件(R3.1)。
- 无 MPI 构建(`make seq`)时 parallel_* 必须回退到顺序参考,用 `#ifdef BPE_HAVE_MPI` 保护。

## 参考实现的语义(必须逐字节复现)

- 每轮选出资格对(至少出现在 2 个不同词里)中前 256 个:频率降序,平局按合并文本 strcmp,再按 pair key(左 token id,再右 token id)。
- 256 个是快照:先选定,再依次合并。入选 pair 即使出现位置已被本轮前面的合并消耗,仍会分配 token id。
- token id:1..255 是字节本身,新 token 按入选顺序从 256 起编号。
- 同一词内的重叠出现,从左到右贪心合并。
- 输出 `token count`,按计数降序,同计数按 strcmp 升序。

## 设计概要

- 把每个 OpenMP 线程视为一个独立 worker,共 16 个,和配置无关。
- 词按哈希分给 worker,每个 worker 独占一部分词。
- 每个 pair 按哈希指定一个所有者 worker,保存全局计数和词数。
- 每轮:各 worker 本地合并 256 个 pair 并记录计数增量 → 增量发给所有者 → 所有者选本地候选 → Allgather → 所有 rank 得出同一份全局前 256。
- 词表(token 文本)在所有 rank 上复制,用于平局比较。
- 线程数由 `BPE_THREADS` 按 rank 号读取,所有 rank 交换线程数得到 worker 编号表。
- 无 MPI 构建下直接调用顺序参考。

## 步骤

每一步完成后验证,再单独提交。

### 步骤 0:合并骨架与测试环境
- 用 skeleton-p2 的文件替换外层 src、tests、Makefile、benchmark.slurm(或在子目录里开发,待定)。
- 本机安装 OpenMPI:`sudo apt-get install openmpi-bin libopenmpi-dev`。
- 写语料生成脚本,覆盖:重叠字符(aaaa)、同文本不同拆法的 pair(ab+c 与 a+bc)、很少的词、词在多个 rank 份额里出现。
- 验证:`make`、`make seq`、`make test` 通过。

### 步骤 1:线程配置
- 每个 rank 读取 `BPE_THREADS`,按 rank 号设置线程数;未设置时用 `OMP_NUM_THREADS`。
- 所有 rank 交换线程数,得到 16 个 worker 的编号表。
- 验证:`mpirun -np 3` 下每个 rank 报告的线程数正确。

### 步骤 2:输入分发
- rank 0 读文件,按线程数比例、在空白处对齐,分块发给各 rank。
- 验证:各 rank 收到的字节数之和等于文件大小,每块都不切词。

### 步骤 3:分布式 Task 1
- 每个 rank 本地计数,再按词哈希把 (词, 计数) 重分布给所有者 worker 去重。
- 验证:全局不同词数和词频总和与顺序版 `task1` 一致。

### 步骤 4:Task 2 单 worker 版
- 先在 1 rank 1 线程下实现「256 个 pair 一批」,带增量记录,先不做通信。
- 验证:输出与 `bin/bpe_seq` 逐字节一致。
- 这一步定下 token id 和平局规则,是最关键的一步。

### 步骤 5:Task 2 多 worker 版
- pair 所有者、增量交换、候选 Allgather、全局选出 256 个。
- 验证:四种配置输出都与 `bin/bpe_seq` 逐字节一致。

### 步骤 6:收尾
- token 计数汇总到 rank 0,只有 rank 0 写 output.txt。
- 检查没有其他 rank 读写文件。
- 验证:随机小语料对拍。

### 步骤 7:性能调优
- 先分段计时:读取、分发、Task 1、每轮合并、通信。
- 针对最慢的部分优化,每次优化后重新验证正确性,每次优化一提交。
- 验证:四种配置在 Spartan 上的耗时。

### 步骤 8:交付物
- 填写 benchmark.slurm 的四个遮住部分:资源请求、模块加载与编译、四个配置元组、srun 启动行。
- README:姓名和学号、Spartan 上的启动命令、四种配置与 1 rank 1 线程自检的复现方法。
- LaTeX 报告(不超过 5000 词):结果、各配置负载不均衡与拖后腿的 rank、读取分发耗时、每轮合并与通信耗时、与 16 线程上限的比较。
- 预约演示。
- 在 Spartan 上完整跑一遍 sbatch。

## 待确认

- 骨架合并方式:覆盖外层目录,还是保留 skeleton-p2 子目录。
- skeleton-p2 里有自己的 `.git`,合并时需要处理,避免嵌套仓库。
- 是否在本机安装 OpenMPI。
