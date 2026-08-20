# K12SimWorld

K12SimWorld is a research prototype for generating executable teaching
simulations from K–12 multimodal questions.

It extends the code-driven reconstruction idea of
[VisPhyWorld](https://github.com/TIGER-AI-Lab/VisPhyWorld) toward educational
problems, while using structured world specifications, declarative physics
backends and trace-based validation inspired by
[EduIllustrate](https://github.com/ECNU-innoSpark/EduIllustrate).


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

## Physics benchmark generation

```bash
cd K12SimWorld
conda activate k12simworld

python run_k12simworld.py generate \
  --benchmark "physics_k12simbench.jsonl" \
  --output-dir "physics_full" \
  --model qwen3-vl-plus \
  --render \
  --jobs 2 \
  --resume \
  --retry-failed
```
