# K12SimWorld GitHub 发布清单

目标仓库：`https://github.com/Lumos000/K12SimWorld`

## 应上传

### 核心源码

- `k12simworld/`：数据整理、筛选、生成、求解、验证和评价；
- `run_k12simworld.py`、`run_k12_screening.py`：公开命令行入口；
- `src/api_config.py`、`src/llm_client.py`：模型配置和调用适配；
- `src/canvas_html_renderer.py`、`src/domain_canvas_renderer.py`、
  `src/threejs_renderer.py`、`src/p5js_renderer.py`、`src/manim_renderer.py`、
  `src/video_normalizer.py`：K12SimWorld 实际依赖的渲染层；
- `assets/threejs/`：离线浏览器渲染资源的唯一权威副本。

### 测试、配置与复现材料

- `tests/`：核心求解器、生成管线、评测、渲染和模型适配层测试；
- `.github/workflows/tests.yml`；
- `.env.template`，只能包含空值或非敏感默认地址；
- `.gitignore`、`requirements-k12.txt`、`package.json`、`package-lock.json`；
- `README.md`、`NOTICE`、`LICENSE`、`CONTRIBUTING.md`、`SECURITY.md`；
- `k12simworld/docs/`、`k12simworld/schemas/`、`k12simworld/examples/`、
  `k12simworld/paper/` 和人工题号清单 `k12simworld/selection/`。

人工题号清单可以公开，因为它用于复现实验选择；其中不应嵌入原始题图、答案或受限数据。

## 不应上传

- `.env`、任何真实 API Key、Token、Cookie 或代理鉴权信息；
- `/path/to/K12-Vista/`、仓库内 `data/` 或其他 K12Vista 原始题目和题图；
- 全量筛选输出、SQLite 检查点、模型原始响应和 API 使用明细；
- `output/`、`runs/`、`results/`、`videos/`、`traces/`、日志和缓存；
- `node_modules/`、Python 环境、`__pycache__/`；
- `external/` 下载内容和批量生成视频；
- `run_*_sub_cuda*.sh` 等包含服务器绝对路径、GPU 编号和付费批处理设置的本地脚本；
- 下载到本地的 `gallery.html` 及其分页图片资源；
- 论文编译产物 `main.pdf`、LaTeX 中间文件。

## 不要直接执行 `git add .`

当前工作目录由 VisPhyWorld 上游仓库演化而来，还包含上游项目网站、评价脚本和本地下载。
向一个全新的 K12SimWorld 仓库发布时，推荐在干净分支或干净克隆中按本页“应上传”清单选择文件。

仓库提供一个只复制公开白名单、不会执行 Git 命令的导出脚本。目标目录必须为空、只包含
`git clone` 创建的 `.git`，或者仅额外包含 GitHub 初始化生成的 `LICENSE`。如果存在该
`LICENSE`，脚本会用同时保留 VisPhyWorld 上游和 K12SimWorld 版权声明的许可证替换它：

```bash
git clone https://github.com/Lumos000/K12SimWorld.git /path/to/K12SimWorld-public
cd /path/to/current/project
bash scripts/export_public_release.sh /path/to/K12SimWorld-public
cd /path/to/K12SimWorld-public
```

如果目标仓库尚无任何提交，`git clone` 仍会创建一个只有 `.git` 的目录，脚本可以正常使用。
脚本明确排除上游旧项目网站、评价视频、本地批处理脚本、数据和所有生成产物。

提交前至少执行：

```bash
git status --short
git diff --cached --stat
git diff --cached --check
git grep -n -I -E 'sk-[A-Za-z0-9_-]{16,}|Bearer [A-Za-z0-9_.-]{16,}' --cached
python -m unittest discover -s tests -p "test_*.py" -v
```

如果密钥曾进入 Git 暂存区或提交历史，仅从文件中删除并不够；应立即吊销旧密钥并重写历史。

## 上游关系

K12SimWorld 使用并修改了 VisPhyWorld 的模型与浏览器渲染基础设施，因此必须同时上传
`LICENSE` 和 `NOTICE`，并在 README 中明确注明上游项目。不要把本项目描述为完全独立、
与 VisPhyWorld 无关的实现。
