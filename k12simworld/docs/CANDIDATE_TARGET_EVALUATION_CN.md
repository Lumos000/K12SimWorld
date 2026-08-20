# 候选终态约束仿真与评测

## 1. 方法边界

默认方法为 `k12simworld_candidate_target`。生成模型只收到题目、题图和无答案元数据，
先产生 `CandidateSolution`，再把自己计算出的结果转换为 `SimulationContract`。K12Vista
标准答案和专家轨迹不进入生成请求，只在生成完成后由评测命令读取。

这不是把起点和终点做线性插值。可信领域求解器必须从初态向前积分或求解；终态、事件和
不变量只用于检查。领域程序首次不满足候选目标时，系统最多使用现有的一次修复额度，且禁止
修改题目给定量、候选答案、目标值和容差。

每道成功的可信领域后端题目应包含：

```text
<problem_id>/
├── storyboard.json
├── solution_spec.json
├── world_spec.json
├── simulation_contract.json
├── program.json
├── traces/SIM_*.json
├── observed_trace.json
├── target_validation.json
├── explanation.md
├── videos/SIM_*.mp4          # 使用 --render 时
└── manifest.json
```

`target_validation.json` 评价的是“仿真是否忠实实现模型自己的候选解答”，不能替代与标准
答案比较得到的物理正确率。

## 2. 环境变量

```bash
cd /root/K12SimWorld
conda activate /root/autodl-tmp/envs/k12simworld

export K12_PHYSICS_DATA=/root/autodl-tmp/data/K12-Vista/experiments/physics_human_v1
export K12_RUN_ROOT=/root/autodl-tmp/k12simworld_runs
```

## 3. 离线检查

不调用模型、不渲染：

```bash
python run_k12simworld.py validate \
  --benchmark "$K12_PHYSICS_DATA/physics_smoke_20.jsonl"

python -m unittest discover -s tests -p 'test_*.py'
```

## 4. 第一题候选终态约束测试

只生成结构和可信 trace，不录制视频：

```bash
python run_k12simworld.py generate \
  --benchmark "$K12_PHYSICS_DATA/physics_smoke_20.jsonl" \
  --output-dir "$K12_RUN_ROOT/candidate_target_smoke_1" \
  --model qwen3-vl-plus \
  --limit 1
```

检查结果：

```bash
python run_k12simworld.py validate \
  --benchmark "$K12_PHYSICS_DATA/physics_smoke_20.jsonl" \
  --artifacts "$K12_RUN_ROOT/candidate_target_smoke_1"

python run_k12simworld.py evaluate \
  --manifests "$K12_RUN_ROOT/candidate_target_smoke_1/manifests.jsonl" \
  --output-dir "$K12_RUN_ROOT/candidate_target_smoke_1/evaluation"
```

重点查看：

- `success_rate`：结构、求解和可选渲染的端到端成功率；
- `candidate_target_evaluable_n`：具有可信 trace 和可执行候选目标的题数；
- `candidate_target_pass_rate`：可信 trace 满足候选终态/不变量的比例；
- `candidate_constraint_satisfaction`：候选约束检查的平均满足率；
- `simulation_correctness`：只有合并专家参考分数后才是金标准仿真正确性。

## 5. 第一题真实视频

使用新目录，避免 `--resume` 跳过之前未渲染的结果：

```bash
python run_k12simworld.py generate \
  --benchmark "$K12_PHYSICS_DATA/physics_smoke_20.jsonl" \
  --output-dir "$K12_RUN_ROOT/candidate_target_smoke_1_render" \
  --model qwen3-vl-plus \
  --limit 1 \
  --render
```

正常题目调用模型三次：候选解答/分镜、WorldSpec/Contract、声明式程序。结构或候选目标
首次失败时最多增加一次修复调用，因此每题最多四次模型调用。

## 6. Smoke 20 与完整集

```bash
python run_k12simworld.py generate \
  --benchmark "$K12_PHYSICS_DATA/physics_smoke_20.jsonl" \
  --output-dir "$K12_RUN_ROOT/candidate_target_smoke_20" \
  --model qwen3-vl-plus \
  --limit 20 \
  --render \
  --resume

python run_k12simworld.py evaluate \
  --manifests "$K12_RUN_ROOT/candidate_target_smoke_20/manifests.jsonl" \
  --output-dir "$K12_RUN_ROOT/candidate_target_smoke_20/evaluation"
```

Smoke 通过后再将 benchmark 换为：

```text
$K12_PHYSICS_DATA/tiers/physics_native_v1.jsonl
$K12_PHYSICS_DATA/tiers/physics_equation_v1.jsonl
$K12_PHYSICS_DATA/tiers/physics_specialized_v1.jsonl
$K12_PHYSICS_DATA/physics_k12simbench.jsonl
```

每个数据层必须使用独立输出目录。

## 7. 与专家参考轨迹比较

专家参考文件遵循 `k12simworld/schemas/expert_reference.schema.json`。轨迹键采用：

```text
<scene_id>.<object_id>.<quantity>
```

例如 `SIM_1.ball.position`、`SIM_1.ball.speed`。生成结束后可直接从 manifest 读取
`observed_trace_path`：

```bash
python run_k12simworld.py score-traces \
  --references /path/to/physics_expert_references.jsonl \
  --manifests "$K12_RUN_ROOT/candidate_target_smoke_20/manifests.jsonl" \
  --output "$K12_RUN_ROOT/candidate_target_smoke_20/expert_trace_scores.jsonl" \
  --strict
```

把专家轨迹分合并进最终报告：

```bash
python run_k12simworld.py evaluate \
  --manifests "$K12_RUN_ROOT/candidate_target_smoke_20/manifests.jsonl" \
  --scores "$K12_RUN_ROOT/candidate_target_smoke_20/expert_trace_scores.jsonl" \
  --output-dir "$K12_RUN_ROOT/candidate_target_smoke_20/evaluation_with_gold"
```

教师/人工评分也可使用 `{"problem_id":"...","scores":{...}}` JSONL，并重复传入
`--scores /path/to/human_scores.jsonl`。后传入文件覆盖相同题目、相同维度的旧分数。

## 8. 结果解释

- 候选目标通过、金标准失败：仿真忠实实现了模型的错误解答，主要是解题错误；
- 候选目标失败：声明式参数、事件、时长或求解映射未实现模型自己的解答；
- 候选目标与金标准都通过：解题结果与可执行物理过程均通过；
- `not_supported`：自由 Three.js/P5.js/Manim 代码没有可信状态 trace，只能做执行、视频和人工评价；
- 失败题仍按主协议计入成功率，并在适用评价维度计零分。
