# K12SimWorld 声明式确定性物理求解器

## 1. 执行协议

第二层和第三层采用“模型声明、可信执行、固定渲染”的协议：

1. 多模态模型根据题目和题图建立 `EduWorldSpec`；
2. 模型输出受约束的 `simulation_spec`，负责对象、拓扑、初值和参数；
3. 项目内求解器执行学科规律并生成逐状态 trace；
4. 固定编译器只根据 trace 绘图，不重新计算物理过程；
5. 校验器重新执行同一 spec 并核对 trace SHA-256；
6. 可选使用 Puppeteer 和 FFmpeg 将可信 Canvas 页面录制为 MP4。

模型仍需完成多模态理解和结构化建模，但不能通过画一条“看起来正确”的轨迹绕过
洛伦兹力、基尔霍夫定律或光线交点计算。


## 2. 第一层：mechanics-2d

Native 力学题默认使用 `mechanics_2d`：模型声明二维物体、线段、力、冲量、弹簧、
定长约束和教学标注；可信求解器输出状态 trace；固定 Canvas 编译器负责绘制。同一
物理对象的左右面板、水平投影或竖直投影使用 `visual_instances`，每个实例通过
`source_object_id` 引用真实 body。视觉实例没有独立物理状态，不能成为力、动作、
碰撞、事件或评测目标。
只有通过空间证据门控的题目才改用 Three.js/Cannon 3D。详见
`ARCHITECTURE_2D_FIRST_CN.md`。

```bash
python run_k12simworld.py simulate-domain \
  --engine mechanics-2d \
  --spec k12simworld/examples/domain/projectile_2d.json \
  --output-dir /tmp/k12-mechanics-2d-demo --render
```

## 3. 第二层：equation-solver

当前包含两种互补执行模型。

`charged_particle_2d` 用于带电粒子轨迹：

- 使用固定步长 Boris 方法积分 `m dv/dt = q(E + v × B)`；
- 支持多个粒子、二维位置/速度、平面外 `Bz` 和平面内 `Ex/Ey`；
- 支持矩形 `field_regions`，展示粒子进入或离开有限电场/磁场区域；
- 输出位置、速度、速率、动能、局部场和区域进入/离开事件；
- 无纯磁场时的速率守恒由离线数值测试检查。

`ode_system` 用于显式方程和耦合过程：

- 使用固定步长四阶 Runge–Kutta（RK4）积分一阶常微分方程组；
- 适合电磁感应中的“导体棒速度—感应电动势—电流—安培力—焦耳热”、
  简谐振动和其他可写成有限维状态方程的题目；
- 支持参数定时变化、条件事件、派生可观测量和多通道曲线；
- 可把状态或可观测量绑定为滑块、仪表、转子或灯泡，动态展示对象变化；
- 表达式由白名单 AST 解释器执行，不使用 `eval`，禁止导入、属性访问、下标和任意代码。

```bash
python run_k12simworld.py simulate-domain \
  --engine equation-solver \
  --spec k12simworld/examples/domain/charged_particle.json \
  --output-dir /tmp/k12-equation-demo

python run_k12simworld.py simulate-domain \
  --engine equation-solver \
  --spec k12simworld/examples/domain/electromagnetic_induction_ode.json \
  --output-dir /tmp/k12-induction-demo
```

这两条命令分别验证带电粒子轨迹和电磁感应耦合方程。模型必须依据题目决定使用哪种
`domain_model`；求解器不会替模型猜测题图中的方向、拓扑或缺失参数。

## 4. 第三层：circuit-solver

当前实现确定性线性直流电路：

- 使用改进节点分析求解 KCL、KVL 和欧姆定律；
- 支持电阻、导线、直流电压源、直流电流源、开关、灯泡、电流表和电压表；
- 支持定时开关以及电压、电流、电阻和额定功率变化；
- 输出节点电势、支路电流、电压、功率、电表示数、灯泡状态和归一化亮度；
- 电流表默认近似零内阻，电压表默认高内阻，题目给出内阻时应显式填写。

```bash
python run_k12simworld.py simulate-domain \
  --engine circuit-solver \
  --spec k12simworld/examples/domain/circuit_switch.json \
  --output-dir /tmp/k12-circuit-demo
```

## 5. 第三层：ray-optics

当前实现二维几何光线追迹：

- 计算射线与有限线段的最近交点；
- 镜面使用向量反射定律；
- 折射界面使用 Snell 定律并检测全反射；
- 薄透镜使用近轴方向变换，可表现平行光线通过焦点；
- 支持屏幕和吸收面，输出每次相交位置及入射/出射方向。

```bash
python run_k12simworld.py simulate-domain \
  --engine ray-optics \
  --spec k12simworld/examples/domain/ray_lens.json \
  --output-dir /tmp/k12-optics-demo
```

以上命令不调用模型，也不启动视频渲染。输出均包含：

```text
simulation_trace.json   # 客观状态或光路
scene.html              # 可直接在浏览器查看的 Canvas 页面
run_summary.json        # 引擎、路径、输入哈希和摘要
```

确认 Node/Puppeteer/FFmpeg 和服务器 I/O 正常后，可增加 `--render` 生成 `scene.mp4`。

## 6. 在人工三层数据上生成

路由现在自动使用：

- `physics_native_v1.jsonl` → 默认 `mechanics-2d`；仅通过空间证据门控时使用 `threejs-cannon`；
- `physics_equation_v1.jsonl` → `equation-solver`；
- 电路题 → `circuit-solver`；
- 光学题 → `ray-optics`。

先取第二层一题，仅生成结构和可信 trace：

```bash
python -u run_k12simworld.py generate \
  --benchmark /path/to/K12-Vista/experiments/physics_human_v1/tiers/physics_equation_v1.jsonl \
  --output-dir /path/to/K12-Vista/experiments/physics_human_v1/runs/equation_structure \
  --model qwen3-vl-plus --limit 1 --resume
```

第三层同时包含电路和光学，自动按 `simulation_type` 路由：

```bash
python -u run_k12simworld.py generate \
  --benchmark /path/to/K12-Vista/experiments/physics_human_v1/tiers/physics_specialized_v1.jsonl \
  --output-dir /path/to/K12-Vista/experiments/physics_human_v1/runs/specialized_structure \
  --model qwen3-vl-plus --limit 1 --resume
```

成功题除 storyboard、WorldSpec 和 program 外，还会包含：

```text
<problem_id>/traces/SIM_*.json
```

## 7. 能力边界

求解器正确不代表模型对题图的建模正确。模型仍可能识别错电路拓扑、正负电荷、场方向、
透镜类型或参数；这些错误会被求解器忠实执行，必须通过题意一致性、关键事件和答案验证评价。

当前明确不覆盖：

- 相对论运动、辐射、自洽粒子—场耦合和任意空间变化的电磁场；
- 交流相量、暂态电容/电感、二极管/晶体管等非线性电路；
- 波动光学、干涉、衍射、偏振、厚透镜和高阶像差；
- 从任意复杂题图自动恢复绝对尺度而无需模型判断。

`equation-solver` 提供的是“可审计方程执行能力”，不等于 105 道 equation 题都能无条件
自动成功：若题目无法给出封闭状态方程、关键参数缺失或需要连续场 PDE，仍应进入人工复核。

遇到这些题目时应标记 `manual_review` 或扩展求解器，不能用装饰动画冒充物理仿真。
