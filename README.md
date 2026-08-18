# K12SimWorld

**Executable and pedagogically grounded simulations for K–12 multimodal problems.**

K12SimWorld turns a K–12 question and its image into an auditable teaching
artifact: a step-by-step explanation, a structured `EduWorldSpec`, a declarative
simulation program, an executed state trace, and an optional video. It evaluates
whether a multimodal model can build and execute a scientifically meaningful
world—not merely produce an attractive animation.

> Research preview. The repository provides the task protocol, data curation
> tools, deterministic physics backends, generation pipeline, evaluation code,
> tests, and an arXiv paper draft. It does not distribute K12Vista data or claim
> measured improvements in real classroom learning outcomes.

中文文档：[实现说明](k12simworld/README_CN.md) ·
[2D-first 架构](k12simworld/docs/ARCHITECTURE_2D_FIRST_CN.md) ·
[候选终态约束与评测](k12simworld/docs/CANDIDATE_TARGET_EVALUATION_CN.md) ·
[物理求解器](k12simworld/docs/DOMAIN_SOLVERS_CN.md) ·
[物理首轮实验](k12simworld/docs/PHYSICS_FIRST_RUN_CN.md) ·
[GitHub 发布清单](docs/REPOSITORY_RELEASE_CN.md)

## What it does

```text
question + image
       │
       ▼
CandidateSolution + pedagogical storyboard
       │
       ▼
EduWorldSpec + SimulationContract
       │
       ▼
2D-first router ──► declarative simulation_spec
                              │
                              ▼
             trusted solver execution + trace replay
                              │
                              ▼
          fixed Canvas / verified 3D / Manim rendering
                              │
                              ▼
 answer, candidate-target, physics and teaching evaluation
```

The model is responsible for understanding the question image, identifying
objects and topology, choosing equations, and supplying justified parameters.
Trusted project code executes the declared model and replays it during
validation. A visually plausible but physically false trajectory therefore
cannot pass trace validation.

## Design principles

- **Candidate-conditioned simulation.** The model first solves the problem; its
  structured answer becomes an auditable simulation contract, never a gold label.
- **2D first.** Native mechanics uses declarative `mechanics-2d`. A model may
  request 3D only with verifiable spatial evidence from the question or caption.
- **Trusted execution.** The model declares objects, parameters and equations;
  project-owned solvers execute them and project-owned compilers generate HTML.
- **Gold-free generation.** Reference answers and expert traces are withheld from
  every generation request and are introduced only during evaluation.
- **Failures stay visible.** Every item writes a manifest, batch generation keeps
  going after per-item failures, and failed items remain in report denominators.

### Render-first validation policy

K12SimWorld distinguishes an unsafe or unexecutable artifact from an imperfect but
renderable artifact. The latter proceeds to rendering with structured warnings:

| Result | Pre-render behavior |
|---|---|
| Model says simulation is unnecessary | Continue; synthesize `SIM_AUTO_1` when needed |
| Top-level and structured final answers differ | Use `solution.final_answer`; record the conflict |
| Adjacent blocks do not alternate | Preserve declared order; record a warning |
| Storyboard and program scene IDs differ | Render available unique scenes; audit missing/extra IDs |
| Auxiliary static geometry is absent from WorldSpec | Continue; keep dynamic-body identity strict |
| Candidate targets are absent or miss tolerance | Preserve the report and continue to rendering |

Hard failures remain for malformed contracts, unsupported engines, unsafe generated
code, duplicate scene IDs, invented dynamic bodies, solver/trace inconsistency and
actual renderer failure after one execution-log-aware repair.

All soft decisions are written to `manifest.diagnostics.pre_render_warnings`; they
do not count as proof of physical correctness.

## Physics execution tiers

| Tier | Backends | Typical processes |
|---|---|---|
| `native` | declarative `mechanics-2d` by default; verified Three.js/Cannon 3D exception | mechanics, collision, pulley, spring, projectile motion |
| `equation` | Boris + restricted-expression RK4 | charged particles, electromagnetic induction, coupled ODE systems |
| `specialized` | MNA circuit solver + geometric ray tracer | switches, meters, lamp brightness, reflection, refraction, lenses |

Current deterministic specialized support includes:

- 2-D charged-particle motion in electric and out-of-plane magnetic fields;
- safe first-order ODE systems with parameters, actions, events and observables;
- linear DC circuits with sources, resistors, wires, switches, lamps, ammeters
  and voltmeters;
- finite-segment ray intersections, reflection, Snell refraction, total internal
  reflection, paraxial thin lenses, screens and absorbers.

See [the solver guide](k12simworld/docs/DOMAIN_SOLVERS_CN.md) for schemas,
examples, numerical methods and explicit limitations.

## Repository layout

```text
k12simworld/                 Core K12 task, pipeline and deterministic solvers
  docs/                      Annotation and experiment protocols
  evaluation/                Metrics, statistics and trace scoring
  examples/                  Small synthetic/public examples
  paper/                     arXiv manuscript source
  schemas/                   Public JSON Schemas
  selection/                 Reproducible ID-only human selection
k12simworld_tests/           Offline unit and integration tests
src/                         Reused/adapted model and rendering infrastructure
assets/threejs/              Local browser-rendering dependencies
run_k12simworld.py           Main CLI
run_k12_screening.py         Full K12Vista screening CLI
```

## Installation

Python 3.10 or newer is recommended.

This repository is private. Authenticate with GitHub CLI without embedding a
personal access token in the remote URL:

```bash
conda install -n base -c conda-forge gh -y
gh auth login --hostname github.com --git-protocol https --web

cd /root
gh repo clone Lumos000/K12SimWorld
cd /root/K12SimWorld

conda create -n k12simworld python=3.10 -y
conda activate k12simworld
python -m pip install --upgrade pip
python -m pip install -r requirements-k12.txt
```

Video rendering additionally requires Node.js 18+, npm, Puppeteer and FFmpeg:

```bash
conda install -c conda-forge nodejs=18 -y
npm ci
node --version
npm --version
node -e "console.log(require.resolve('puppeteer'))"
ffmpeg -version
```

The deterministic solvers and tests do not require a model API or video rendering.

## Offline quick start

Run the complete test suite:

```bash
python -m unittest discover -s k12simworld_tests -p "test_*.py" -v
```

Run a declarative 2-D projectile end to end without an LLM:

```bash
python run_k12simworld.py simulate-domain \
  --engine mechanics-2d \
  --spec k12simworld/examples/domain/projectile_2d.json \
  --output-dir output/projectile-demo --render
```

Generate an auditable charged-particle trace and self-contained Canvas page:

```bash
python run_k12simworld.py simulate-domain \
  --engine equation-solver \
  --spec k12simworld/examples/domain/charged_particle.json \
  --output-dir output/charged-particle-demo
```

Other offline examples:

```bash
python run_k12simworld.py simulate-domain \
  --engine equation-solver \
  --spec k12simworld/examples/domain/electromagnetic_induction_ode.json \
  --output-dir output/induction-demo

python run_k12simworld.py simulate-domain \
  --engine circuit-solver \
  --spec k12simworld/examples/domain/circuit_switch.json \
  --output-dir output/circuit-demo

python run_k12simworld.py simulate-domain \
  --engine ray-optics \
  --spec k12simworld/examples/domain/ray_lens.json \
  --output-dir output/optics-demo
```

Each command writes `simulation_trace.json`, `scene.html`, and
`run_summary.json`. Add `--render` only after the local Node/Puppeteer/FFmpeg
preflight succeeds.

## K12Vista data

K12Vista is not included. Download it from the
[official Hugging Face dataset](https://huggingface.co/datasets/lipku1999/K12-Vista)
and review its license and the [dataset paper](https://arxiv.org/abs/2506.01676).
Keep the raw JSONL and decoded images outside Git, for example:

```text
data/K12-Vista/raw/K12_Vista.jsonl
```

The included `.gitignore` excludes `data/`, generated screening results,
checkpoints, raw model responses, traces and videos.

## Model configuration

```bash
cp .env.template .env
chmod 600 .env
```

For Qwen through Alibaba Cloud Model Studio, set only the matching regional key
and endpoint in `.env`:

```dotenv
DASHSCOPE_API_KEY="your-key"
DASHSCOPE_API_BASE="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
```

The Beijing, Singapore and Virginia endpoints use different regional keys; do
not mix them. `qwen3-vl-plus` is selected at the CLI with `--model`, not stored
as a required global default.

Fill only the providers you use. Never commit `.env`. Full-dataset screening
uses `DEEPSEEK_TEXT_MODEL` for high-recall text routing and `QWEN_VL_MODEL` for
multimodal final decisions. Generation can use the model adapters supported by
`src/llm_client.py`.

## Full-dataset screening

The screening pipeline is checkpointed and resumable. Start with the local
audit and five-item API smoke test before paid full-dataset calls:

```bash
python run_k12_screening.py \
  --input data/K12-Vista/raw/K12_Vista.jsonl \
  --output output/k12vista-selected \
  --phase audit

python run_k12_screening.py \
  --input data/K12-Vista/raw/K12_Vista.jsonl \
  --output output/k12vista-selected \
  --phase smoke
```

After both model endpoints pass, `--phase all` resumes cached successes and
runs text routing, multimodal adjudication, stratified audits, finalization and
the paginated human-review gallery:

```bash
python run_k12_screening.py \
  --input data/K12-Vista/raw/K12_Vista.jsonl \
  --output output/k12vista-selected \
  --phase all
```

Do not commit the resulting records: they may contain dataset text, images,
model responses and provider usage metadata.

## Main generation workflow

Set the external dataset and run roots used in the physics experiments:

```bash
cd /root/K12SimWorld
conda activate /root/autodl-tmp/envs/k12simworld

export K12_PHYSICS_DATA=/root/autodl-tmp/data/K12-Vista/experiments/physics_human_v1
export K12_RUN_ROOT=/root/autodl-tmp/k12simworld_runs
```

Validate benchmark records before making paid model calls:

```bash
python run_k12simworld.py validate \
  --benchmark "$K12_PHYSICS_DATA/physics_smoke_20.jsonl"
```

Start with one item and omit video rendering:

```bash
python run_k12simworld.py generate \
  --benchmark "$K12_PHYSICS_DATA/physics_smoke_20.jsonl" \
  --output-dir "$K12_RUN_ROOT/smoke_1_structure" \
  --model qwen3-vl-plus \
  --limit 1
```

Render the frozen eight-item representative set. Batch generation records each
failure and continues with the next problem:

```bash
export REP_DATA="$K12_PHYSICS_DATA/representative/physics_representative_8_v1.jsonl"
export REP_RUN="$K12_RUN_ROOT/representative_8_relaxed"

python run_k12simworld.py generate \
  --benchmark "$REP_DATA" \
  --output-dir "$REP_RUN" \
  --model qwen3-vl-plus \
  --limit 8 \
  --render \
  --resume \
  --retry-failed
```

After representative tests are stable, render all 433 selected physics items:

```bash
export FULL_DATA="$K12_PHYSICS_DATA/physics_k12simbench.jsonl"
export FULL_RUN="$K12_RUN_ROOT/physics_full_relaxed"

python run_k12simworld.py generate \
  --benchmark "$FULL_DATA" \
  --output-dir "$FULL_RUN" \
  --model qwen3-vl-plus \
  --render \
  --resume \
  --retry-failed
```

Each problem writes an auditable directory:

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
├── videos/SIM_*.mp4          # only with --render
└── manifest.json
```

Trusted domain engines emit state traces and candidate-target reports. Free-code
Three.js/P5.js/Manim scenes can still be rendered, but do not claim a trusted
physics trace.

`--resume` skips every existing manifest. Adding `--retry-failed` reruns only
existing failures while retaining successes. Use a new output directory when
changing prompts, models, validation policy or whether `--render` is enabled;
otherwise an old successful manifest will be skipped.

Failed items remain in `manifests.jsonl` and in evaluation denominators rather
than being silently discarded.

## Evaluation

Evaluation keeps solution, simulation and pedagogical quality separate. Failed
generations remain in the denominator and receive zero rather than being
removed from the report. Candidate-target scores measure whether the simulation
implements the model's own answer; they do not replace gold-answer, expert
physics or teacher-rated pedagogical evaluation:

```bash
python run_k12simworld.py evaluate \
  --manifests "$REP_RUN/manifests.jsonl" \
  --output-dir "$REP_RUN/evaluation"
```

Evaluate the complete physics run in the same way:

```bash
python run_k12simworld.py evaluate \
  --manifests "$FULL_RUN/manifests.jsonl" \
  --output-dir "$FULL_RUN/evaluation"
```

The report separates end-to-end success, candidate-target pass rate, execution
quality, answer correctness, physical correctness and pedagogical usefulness.

Post-generation contract validation is intentionally stricter than the
render-first gate: it reports non-passing candidate targets without deleting or
hiding rendered artifacts.

```bash
python run_k12simworld.py validate \
  --benchmark "$FULL_DATA" \
  --artifacts "$FULL_RUN"
```

Expert reference traces can be compared with executed observations through
`score-traces`. Human annotation dimensions and statistical reporting are
defined in [the annotation protocol](k12simworld/docs/ANNOTATION_PROTOCOL_CN.md)
and [experiment protocol](k12simworld/docs/EXPERIMENT_PROTOCOL_CN.md).

## Data and result policy

Do not upload raw K12Vista records, images, answer text, API request/response
logs, checkpoints, generated videos, or provider usage files. The exact public
allowlist and pre-push checks are documented in
[GitHub 发布清单](docs/REPOSITORY_RELEASE_CN.md).

## Scope and limitations

- The DC solver does not currently model AC phasors, capacitor/inductor
  transients, diodes or transistors.
- The ray tracer is geometric/paraxial; it does not model interference,
  diffraction, polarization or high-order aberration.
- The equation solver handles prescribed fields and finite-dimensional ODEs,
  not relativistic radiation, self-consistent particle-field coupling or PDEs.
- Backend availability does not guarantee that a model inferred the correct
  image topology, signs, units or parameters.
- Student learning gains require a separate classroom or pre/post-test study.

## Upstream and license

K12SimWorld adapts the execution and rendering infrastructure of
[VisPhyWorld](https://github.com/TIGER-AI-Lab/VisPhyWorld), licensed under MIT.
The upstream copyright and MIT license are preserved in [LICENSE](LICENSE),
with attribution in [NOTICE](NOTICE).

K12SimWorld-specific code is released under the same MIT License unless noted
otherwise.

## Citation

The K12SimWorld manuscript is currently a draft under
[`k12simworld/paper`](k12simworld/paper). A canonical BibTeX entry will be added
after the arXiv identifier and author list are finalized. Until then, please
cite the K12Vista dataset and VisPhyWorld upstream work referenced in the
manuscript rather than inventing a K12SimWorld publication record.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) before changing a solver, schema or
evaluation contract. Report credential exposure or unsafe generated-code
execution according to [SECURITY.md](SECURITY.md), never in a public issue with
the secret attached.
