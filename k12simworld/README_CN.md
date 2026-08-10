# K12SimWorld 实现说明

K12SimWorld 将 VisPhyWorld 的“可执行假设”迁移到 K12 多模态教学内容生成。它不会把 K12Vista 的参考答案发送给被测模型，而是生成以下可审计产物：

1. 图文—模拟交错的教学 storyboard；
2. 单一、不可变的 `EduWorldSpec`；
3. 绑定该规范哈希的独立可执行场景；
4. 可选的 MP4、Markdown 讲解和运行 manifest；
5. 将失败计为零分的评测与论文表格。

## 目录

- `models.py`：K12 问题、storyboard、EduWorldSpec 和 manifest 契约。
- `curation.py`：动态适用性筛选、240 题分层抽样、60 题专家子集和知识点隔离划分。
- `pipeline.py`：教学规划 → WorldSpec → 状态锚定程序 → 一次修复 → 可选渲染。
- `routing.py`：刚体、方程、电路、光学、P5.js 和 Manim 的确定性路由。
- `domain_solvers.py`：带电粒子、受限 ODE、直流电路和二维几何光学可信求解器。
- `domain_compiler.py`：把声明式场景执行为 trace，并编译成固定 Canvas 教学页面。
- `evaluation/`：三类综合分、成功率、人工一致性、置信区间和论文表格。
- `schemas/`：公开 JSON Schema。

## 离线检查

以下命令只处理仓库内样例，不调用模型、不渲染：

```bash
python run_k12simworld.py curate \
  --input k12simworld/examples/sample_k12vista.jsonl \
  --output-dir /tmp/k12simbench-demo \
  --physics-target 2 --extension-target 3 --expert-target 2

python run_k12simworld.py prepare-prompts \
  --benchmark /tmp/k12simbench-demo/k12simbench.jsonl \
  --output-dir /tmp/k12simbench-prompts

python run_k12simworld.py evaluate \
  --manifests k12simworld/examples/evaluation_records.jsonl \
  --output-dir /tmp/k12simbench-results
```

专家参考和执行环境导出的状态轨迹可通过 `score-traces` 对齐；该命令计算初始/最终状态、事件 F1、约束满足率与归一化轨迹 RMSE。轨迹必须来自实际执行或专家记录，不能用模型自报状态冒充执行结果。

## 接入真实 K12Vista

数据入口接受 JSON 或 JSONL，兼容 K12Vista/EduIllustrate 的 `hash_id`、`img`、`question`、`format_answer`、`img_caption`、`difficulty`、`type`、`subject` 和 `knowledge_point` 字段。先将 Hugging Face 数据导出为 JSONL，再运行 `curate`。数据下载和许可审核由研究者单独完成，本仓库不会自动抓取数据。

人工筛选题号可直接冻结为实验清单：

```bash
python run_k12simworld.py prepare-human-selection \
  --input /path/to/K12-Vista/raw/K12_Vista.jsonl \
  --selection k12simworld/selection/physics_human_selection_v1.json \
  --screening-results /path/to/K12-Vista/selected/screening_results.jsonl \
  --output-dir /path/to/K12-Vista/experiments/physics_human_v1 \
  --smoke-target 20 --seed 2026
```

当前人工物理集的分阶段运行方法和预期产物见
`docs/PHYSICS_FIRST_RUN_CN.md`；第二、三层求解器见 `docs/DOMAIN_SOLVERS_CN.md`。

`curate` 输出：

- `k12simbench.jsonl`：筛选结果、知识点隔离 split 和专家子集标记；
- `expert_subset_ids.json`：专家标准模拟子集；
- `rejected.jsonl`：拒绝样本及原因，便于审计；
- `curation_summary.json`：实际分布和配额不足警告。

## 调用模型

模型调用复用项目现有 `src.llm_client.LLMClient`，因此沿用其环境变量。默认不渲染：

```bash
python run_k12simworld.py generate \
  --benchmark /path/to/k12simbench.jsonl \
  --output-dir output/k12simworld/gpt5 \
  --model gpt-5
```

确认本地 Node、FFmpeg、Puppeteer 或 Manim 环境后，显式增加 `--render`。也可用 `--engine` 固定引擎做消融。生成失败后只允许一次完整替换式修复；再次失败会保留错误并计为失败，不生成伪造的空场景。

四个对照系统使用统一入口，例如：

```bash
python run_k12simworld.py generate-baseline \
  --benchmark /path/to/k12simbench.jsonl \
  --output-dir output/k12simworld/direct-code \
  --model gpt-5 --method direct_code
```

将 `direct_code` 替换为 `text_cot`、`static_manim` 或 `unanchored` 即可。每个入口都写出同构 manifest，便于按题目 ID 配对。

## 实验记录要求

每个系统都使用同一题目 ID，`method` 固定为：

- `text_cot`
- `static_manim`
- `direct_code`
- `unanchored`
- `k12simworld_state_anchored`

Judge 或人工评分写入 manifest 的 `scores`。评分采用 0–100；0–1 会自动换算。失败项目的全部维度强制为 0。不要删除失败样本。

文本方法没有模拟总分，缺失模态显示为 `--`，不会用较少维度计算一个看似更高的总体分；但其解题维度失败仍按 0 计。静态图方法同理，不伪造轨迹或动力学分数。

人工标注流程、指标定义和结果陈述边界见 `docs/`，论文模板见 `paper/`。
