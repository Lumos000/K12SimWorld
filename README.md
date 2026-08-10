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
[物理求解器](k12simworld/docs/DOMAIN_SOLVERS_CN.md) ·
[物理首轮实验](k12simworld/docs/PHYSICS_FIRST_RUN_CN.md) ·
[GitHub 发布清单](docs/REPOSITORY_RELEASE_CN.md)

## What it does

```text
question + image
       │
       ▼
pedagogical storyboard
       │
       ▼
EduWorldSpec ──► declarative simulation_spec
                         │
                         ▼
                 deterministic execution
                         │
                         ▼
        state trace + Canvas/Manim/Three.js output
                         │
                         ▼
       answer, simulation and teaching evaluation
```

The model is responsible for understanding the question image, identifying
objects and topology, choosing equations, and supplying justified parameters.
Trusted project code executes the declared model and replays it during
validation. A visually plausible but physically false trajectory therefore
cannot pass trace validation.

## Physics execution tiers

| Tier | Backends | Typical processes |
|---|---|---|
| `native` | Three.js + Cannon.js | mechanics, collision, pulley, spring, projectile motion |
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

```bash
git clone https://github.com/Lumos000/K12SimWorld.git
cd K12SimWorld

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-k12.txt
```

Video rendering additionally requires Node.js 18+, npm, Puppeteer and FFmpeg:

```bash
npm ci
node --version
ffmpeg -version
```

The deterministic solvers and tests do not require a model API or video rendering.

## Offline quick start

Run the complete test suite:

```bash
python -m unittest discover -s k12simworld_tests -p "test_*.py" -v
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
```

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

Prepare or validate a benchmark before calling a paid model:

```bash
python run_k12simworld.py validate --benchmark path/to/k12simbench.jsonl
```

Start with one item and omit video rendering:

```bash
python -u run_k12simworld.py generate \
  --benchmark path/to/physics_equation_v1.jsonl \
  --output-dir output/equation-smoke \
  --model qwen3-vl-plus \
  --limit 1 \
  --resume
```

Successful items contain the storyboard, immutable world specification,
declarative program, executed trace, validation report and manifest. Failed
items remain in evaluation and are not silently discarded.

## Evaluation

Evaluation keeps solution, simulation and pedagogical quality separate. Failed
generations remain in the denominator and receive zero rather than being
removed from the report:

```bash
python run_k12simworld.py evaluate \
  --manifests output/equation-smoke/manifests.jsonl \
  --output-dir output/equation-smoke-evaluation
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
