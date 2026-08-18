# 人工筛选物理题：首轮端到端运行指南

## 已冻结的数据

- 人工选择配置：`k12simworld/selection/physics_human_selection_v1.json`
- 全量物理集：`/path/to/K12-Vista/experiments/physics_human_v1/physics_k12simbench.jsonl`
- 首轮工程烟雾集：`/path/to/K12-Vista/experiments/physics_human_v1/physics_smoke_20.jsonl`
- 完整性与分布：同目录下 `physics_selection_report.json`
- 数据与配置校验和：同目录下 `checksums.json`

Notion 中有 437 次题号记录，按 ID 去重后为 433 题；433 题全部命中
K12Vista 且全部为物理。全量集包含十二年级 342 题、九年级 91 题；问答题 300、
选择题 125、填空题 8。Smoke 集只用于验证工程链路，不用于论文主结果。

标准答案和解析保存在 benchmark 中供后续评价，但 `K12Problem.model_payload()` 会将它们
从被测模型请求中删除。`physics_model_input_preview.jsonl` 是可审计的无答案输入预览；
真实模型请求还会由适配器单独附加原始题图。

## 0. 渲染依赖与服务器负载检查

视频渲染环境需要安装 Node.js、npm、FFmpeg 和 Puppeteer；共享服务器发生磁盘拥塞时，
二进制启动仍可能长时间处于 `D` 状态。
每次实验前检查实际可用性，不要仅凭包记录判断：

```bash
conda activate visphyworld
cd /path/to/K12SimWorld

node --version
npm --version
node -e "console.log(require.resolve('puppeteer'))"
ffmpeg -version | head -n 1
```

同时用 `vmstat 1` 和 `cat /proc/pressure/io` 检查共享磁盘压力。不要在 Node/FFmpeg
版本检查卡住或 I/O PSI 很高时执行 `npm ci`、批量生成或视频渲染。

Equation/circuit/ray-optics 求解器本身不依赖 Node；只有把可信 Canvas HTML 录制为
MP4 时才需要 Node、Puppeteer 和 FFmpeg。详见 `DOMAIN_SOLVERS_CN.md`。

## 1. 离线验证

```bash
cd /path/to/K12SimWorld
conda activate visphyworld

python run_k12simworld.py validate \
  --benchmark /path/to/K12-Vista/experiments/physics_human_v1/physics_smoke_20.jsonl
```

预期输出是 `validation failures: 0`。该步骤不调用 API。

## 2. 先验证模型的三阶段 JSON 生成

先只运行第一题且不渲染。当前 smoke 第一题是标准水平抛体问题，适合定位接口和 JSON
契约错误：

```bash
python run_k12simworld.py generate \
  --benchmark /path/to/K12-Vista/experiments/physics_human_v1/physics_smoke_20.jsonl \
  --output-dir /path/to/K12-Vista/experiments/physics_human_v1/runs/qwen3-vl-plus-structure \
  --model qwen3-vl-plus \
  --limit 1
```

正常情况下每题调用模型三次：storyboard、EduWorldSpec、program；若 program 首次验证失败，
只允许一次修复，因此最多四次。成功题目录应包含：

```text
<problem_id>/
├── storyboard.json
├── world_spec.json
├── program.json
├── explanation.md
└── manifest.json
manifests.jsonl
```

此阶段 `manifest.success` 应为 `true`，但 `video_paths` 为空。失败时先看
`manifest.error`，不要删除失败项，也不要把空白场景当成成功。

## 3. 单题真实渲染

依赖检查通过后，使用一个新的输出目录运行第一题：

```bash
python run_k12simworld.py generate \
  --benchmark /path/to/K12-Vista/experiments/physics_human_v1/physics_smoke_20.jsonl \
  --output-dir /path/to/K12-Vista/experiments/physics_human_v1/runs/qwen3-vl-plus-render \
  --model qwen3-vl-plus \
  --limit 1 \
  --render
```

预期新增 `<problem_id>/videos/SIM_*.mp4` 和对应渲染日志。再执行：

```bash
python run_k12simworld.py validate \
  --benchmark /path/to/K12-Vista/experiments/physics_human_v1/physics_smoke_20.jsonl \
  --artifacts /path/to/K12-Vista/experiments/physics_human_v1/runs/qwen3-vl-plus-render
```

## 4. 从第 1 题断点续跑到 20 题

不要删除第一题。`--resume` 会读取逐题 manifest，跳过已有结果：

```bash
python run_k12simworld.py generate \
  --benchmark /path/to/K12-Vista/experiments/physics_human_v1/physics_smoke_20.jsonl \
  --output-dir /path/to/K12-Vista/experiments/physics_human_v1/runs/qwen3-vl-plus-render \
  --model qwen3-vl-plus \
  --limit 20 \
  --render \
  --resume
```

已有失败也会被保留并跳过。只有明确决定再次付费重试时才增加
`--retry-failed`。最终 `manifests.jsonl` 应恰好有 20 条，不应按成功与否删除题目。

## 5. 评价与预期结果边界

生成 manifest 的金标准/人工 `scores` 默认仍为空；可信领域后端会额外写出候选解答、
仿真契约、观测 trace 和候选目标验证结果。直接运行 `evaluate` 可可靠报告执行成功率、
候选目标通过率与成本，但金标准解题/物理/教学质量分仍显示 `--`：

```bash
python run_k12simworld.py evaluate \
  --manifests /path/to/K12-Vista/experiments/physics_human_v1/runs/qwen3-vl-plus-render/manifests.jsonl \
  --output-dir /path/to/K12-Vista/experiments/physics_human_v1/results/qwen3-vl-plus-smoke
```

论文级质量结果还需要把自动 Judge、专家或教师评分写入 manifest 的 `scores`，维度与量表见
`ANNOTATION_PROTOCOL_CN.md`。客观轨迹分数还需要专家参考轨迹和执行器导出的观测轨迹；
candidate-target 方法的可信领域后端会写出 `observed_trace.json`，自由代码后端仍只有视频，
不能把 MP4 冒充状态轨迹。生成、`score-traces` 与 `evaluate --scores` 的完整命令见
`CANDIDATE_TARGET_EVALUATION_CN.md`。

因此首轮 smoke 的合理成功标准是：

1. 20 条输入均产生 manifest，失败项没有被遗漏；
2. 成功项具有合法 storyboard、WorldSpec 和绑定同一 WorldSpec 哈希的 program；
3. 渲染成功项具有可播放 MP4，且日志中无网络资源依赖；
4. 候选答案与 K12Vista 标准答案分离生成，之后才做盲评；
5. 报告真实成功率、一次修复率、耗时和 token，不预设质量数字。

不能预期或预写“20 题全部成功”“提高学习效果”等结论。若只完成视频生成，只能说明工程
链路跑通；只有答案正确性、关键事件/状态和教师评分共同改善，才能讨论教育推理收益。
