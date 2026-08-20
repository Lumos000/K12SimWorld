# K12SimWorld

K12SimWorld is a research prototype for generating executable teaching
simulations from K–12 multimodal questions.

It extends the code-driven reconstruction idea of
[VisPhyWorld](https://github.com/TIGER-AI-Lab/VisPhyWorld) toward educational
problems, while using structured world specifications, declarative physics
backends and trace-based validation inspired by
[EduIllustrate](https://github.com/ECNU-innoSpark/EduIllustrate).

> Work in progress: interfaces, prompts, solvers and evaluation protocols may
> still change. The repository does not include K12Vista data and should not yet
> be treated as a production teaching system.

## Current workflow

```text
question + image
  -> candidate solution
  -> pedagogical storyboard
  -> EduWorldSpec
  -> declarative simulation program
  -> trusted execution trace
  -> optional video rendering
  -> physics and teaching evaluation
```

Current capabilities include:

- 2D-first engine routing with a controlled 3D exception;
- declarative mechanics, equation, circuit and ray-optics backends;
- distance, path/curve and event-triggered constraints;
- candidate-target and physical trace validation;
- per-item failure isolation, retry, resume and parallel generation;
- deterministic Canvas frame capture and optional Three.js/Cannon.js rendering.

## Installation

Python 3.10 is recommended.

```bash
cd /root/K12SimWorld
conda create -n k12simworld python=3.10 -y
conda activate k12simworld
python -m pip install --upgrade pip
python -m pip install -r requirements-k12.txt
```

Video rendering additionally needs Node.js 18+, Puppeteer and FFmpeg:

```bash
conda install -c conda-forge nodejs=18 ffmpeg -y
npm ci
```

Configure only the model provider you use:

```bash
cp .env.template .env
chmod 600 .env
```

For Qwen/DashScope, set the corresponding regional API key and compatible API
endpoint in `.env`. Never commit `.env`.

## Tests

```bash
conda activate /root/autodl-tmp/envs/k12simworld
cd /root/K12SimWorld
python -m unittest discover -s k12simworld_tests
```

Run an offline declarative example:

```bash
python run_k12simworld.py simulate-domain \
  --engine mechanics-2d \
  --spec k12simworld/examples/domain/pendulum_break_2d.json \
  --output-dir output/pendulum-demo \
  --render
```

## Physics benchmark generation

Set the external data and output locations:

```bash
export K12_PHYSICS_DATA=/root/autodl-tmp/data/K12-Vista/experiments/physics_human_v1
export K12_RUN_ROOT=/root/autodl-tmp/k12simworld_runs
```

Start with the representative subset:

```bash
python run_k12simworld.py generate \
  --benchmark "$K12_PHYSICS_DATA/representative/physics_representative_8_v1.jsonl" \
  --output-dir "$K12_RUN_ROOT/representative_8" \
  --model qwen3-vl-plus \
  --limit 8 \
  --render \
  --jobs 2 \
  --resume \
  --retry-failed
```

Run the complete selected physics benchmark in a new output directory whenever
prompts or solver behavior change:

```bash
export K12SIMWORLD_RENDER_MODE=auto
export K12SIMWORLD_CAPTURE_FPS=5
export K12SIMWORLD_BROWSER_GPU=auto
export K12SIMWORLD_FFMPEG_ENCODER=libx264

python run_k12simworld.py generate \
  --benchmark "$K12_PHYSICS_DATA/physics_k12simbench.jsonl" \
  --output-dir "$K12_RUN_ROOT/physics_full_path_constraints_v21" \
  --model qwen3-vl-plus \
  --render \
  --jobs 2 \
  --resume \
  --retry-failed
```

Do not force one engine for the full benchmark; the router selects mechanics,
equation, circuit, optics or verified 3D execution according to the problem.

## Evaluation

```bash
python run_k12simworld.py evaluate \
  --manifests "$K12_RUN_ROOT/physics_full_path_constraints_v21/manifests.jsonl" \
  --output-dir "$K12_RUN_ROOT/physics_full_path_constraints_v21/evaluation"
```

Generation success only means that an artifact was produced. Physical validity,
answer correctness and pedagogical usefulness are reported separately.

## Repository layout

```text
k12simworld/          pipeline, schemas, solvers and evaluation
k12simworld_tests/    offline unit and integration tests
src/                  model and rendering infrastructure
assets/threejs/       browser rendering assets
run_k12simworld.py    main CLI
```

## Known limitations

The mechanics backend is intentionally bounded and is not a general rigid-body
solver. Complex tipping, persistent multi-body friction, flexible bodies and
unsupported spatial interactions may require a specialized backend or a
verified Three.js/Cannon.js scene. Generated videos should therefore be reviewed
with their execution traces and validation reports.
