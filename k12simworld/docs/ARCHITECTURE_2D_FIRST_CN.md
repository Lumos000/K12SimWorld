# K12SimWorld 2D-first 可审计架构

## 目标

Native 力学题默认不再让模型生成 HTML/JavaScript。模型只负责理解题目并声明场景；可信后端负责执行、绘制与录像。3D 是需要证据的例外，不是视觉风格选项。

## 完整流程与职责

```text
题目 + 题图
  │
  ├─ 1. Storyboard 模型
  │     解题；选择教学步骤；默认申请 schematic_2d；必要时提交受控 3D 证据
  │
  ├─ 2. 空间证据门控（可信代码）
  │     核对 criterion；核对 evidence_quote 是否逐字来自题干/图注；失败则强制 2D
  │
  ├─ 3. EngineRouter（可信代码）
  │     mechanics-2d / equation-solver / circuit-solver / ray-optics / verified Three.js 3D
  │
  ├─ 4. EduWorldSpec 模型
  │     统一对象 ID、坐标、单位、参数、约束、初末状态和预期事件
  │
  ├─ 5. SimulationSpec 模型
  │     声明真实 bodies、物理关系、连续 phases、事件动作和默认单一完整场景
  │
  ├─ 6. mechanics-2d 求解器（可信代码）
  │     固定步长积分、约束反力、位置/速度投影、接触、碰撞、弹簧和断绳事件；生成 trace
  │
  ├─ 7. 校验器（可信代码）
  │     重放 spec 并核对 trace 哈希；检查 WorldSpec 对象覆盖、单位、场景和安全契约
  │
  ├─ 8. Canvas 编译器（可信代码）
  │     固定绘制物体、轨迹、速度/合力箭头、弹簧、绳和标签；不重新计算物理
  │
  └─ 9. Puppeteer + FFmpeg
        可信 Canvas 按指定物理进度离线抽帧；自由代码实时录制；ffprobe 检查真实 MP4
```

## 3D 申请协议

Storyboard 可返回：

```json
{
  "visualization_decision": {
    "mode": "spatial_3d",
    "criterion": "non_coplanar_motion",
    "evidence_quote": "不共面三维运动",
    "reason": "轨迹不能在单一平面表达"
  }
}
```

允许的 criterion 只有：

- `non_coplanar_motion`
- `depth_dependent_collision`
- `spatial_rotation_axis`
- `perspective_geometry_required`
- `occlusion_is_physics`
- `multi_view_spatial_structure`

系统要求 evidence_quote 是题干或图注中的精确子串，并含有与 criterion 对应的空间词。普通透视插画、“3D 更好看”或物体在现实中是立体的，都不能通过。CLI 显式 `--engine threejs-cannon` 仍作为用户授权的实验覆盖项保留，并在审计信息中标注为 override。

## mechanics-2d v1 能力边界

支持：圆、矩形、杆；动态/静态/运动学物体；有限线段地面和斜面；恒力；定时冲量/位置/速度；弹簧；含反作用力及速度级修正的定长绳；按时间或位置穿越/速度阈值触发的断开与恢复约束；基础物体碰撞；连续阶段、事件快照、轨迹、速度和合力标注。

默认使用 `visual_strategy="continuous_process"` 和空 `visual_instances`，在同一画布中同时显示支点、绳、物体、环境、轨迹和事件后的后续运动。`phases` 只负责标注连续过程，不改变物理状态；断绳必须使用 `remove_distance_constraint`，不能只写在旁白里。

只有题目明确要求速度分解或坐标投影时，才使用
`visual_strategy="component_decomposition"` 和 `visual_instances`：

```json
{"id":"ball_h","source_object_id":"ball",
 "view":"horizontal_projection","panel":"right","show_trail":true}
```

`ball_h` 只是一份由 `ball` trace 派生的视觉实例，不进入积分、受力、碰撞、事件、
invariant 或 target observable。禁止为了左右分面而在 `bodies` 中复制 `ball_h` /
`ball_v`；真实 `bodies` 和 `static_geometry` 仍必须逐一存在于 EduWorldSpec。

暂不覆盖：精确刚体角冲量、多点接触堆叠、复杂滑轮拓扑、圆弧/样条碰撞形状、断簧、连续碰撞检测。当前断绳覆盖“移除一条定长约束并连续保留瞬时位置/速度”；需要绳质量、弹性波或复杂滑轮联动时仍应扩展求解器或标记人工复核，不能退回装饰动画冒充物理仿真。

## 渲染时钟与吞吐

可信求解器已经产生完整 trace，因此视频时钟不再依赖墙钟。Canvas 编译器公开
`window.__k12simRenderFrame(progress)`；快速捕获器依次设置归一化进度并立即截图，
再由 FFmpeg 编成固定 5 fps 视频。这既缩短耗时，也保证断绳、碰撞等关键事件不会因
`requestAnimationFrame` 卡顿而漏帧。`K12SIMWORLD_RENDER_MODE=realtime` 只用于
回归比较；自由 Three.js/P5.js 因缺少可信帧接口，仍使用 MediaRecorder。

浏览器 GPU 与 CUDA 是不同通道。`K12SIMWORLD_BROWSER_GPU=auto` 会尝试硬件合成，
并允许 SwiftShader 回退；日志中的实际 WebGL renderer 才是是否使用 GPU 的证据。
二维求解、trace 编译和 512×512 PNG 编码主要受 CPU/进程启动影响，GPU 不是主要
加速点。批量吞吐由 CLI `--jobs` 在题目级并发提升，每个 worker 拥有独立模型客户端；
远程 API 首轮建议 `--jobs 2`。

## 可审计产物

每道题保存：

- `storyboard.json`：模型申请和 `visualization_audit`
- `world_spec.json`：不可变世界状态
- `program.json`：声明式 spec、可信 trace、trace SHA-256 和 solver version
- `traces/SIM_*.json`：物理状态序列
- `videos/SIM_*.mp4`：固定渲染视频
- `manifest.json`：有效模式、引擎、空间审计、调用次数和错误
