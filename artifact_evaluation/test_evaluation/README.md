# Tiny Artifact Evaluation

This directory mirrors the full artifact-evaluation workflow, but it uses only
the microbenchmarks under `checkpoint_eval/models/microbenchmarks` and writes
small PNG/CSV outputs. It is intended as a quick smoke test for generating the
same four figure shapes at a much smaller scale.

The tiny Figure 8, Figure 9, and Figure 11 scripts include CheckFreq, GPM,
Gemini, and PCcheck. The tiny Figure 12 script is PCcheck-only because it tests
PCcheck async checkpointing.

These figures are microbenchmark-derived proxies, not paper-scale reproduction
results. For paper-scale runs, use `artifact_evaluation/README.md`.

## Setup

Use the same PCcheck installation as the full artifact evaluation:

```bash
cd pccheck
bash install.sh
```

You do not need to run `setup_models_and_datasets.sh` for this tiny evaluation.

Optional environment variables:

```bash
export PCCHECK_PYTHON=python3.9
export PCCHECK_LIB_PATH=$HOME/pccheck/checkpoint_eval/pccheck/libtest_ssd.so
```

If these are not set, the scripts use the Python executable that launched them
and the in-repo `checkpoint_eval/pccheck/libtest_ssd.so` path.

Gemini defaults to a single-rank microbenchmark on the local VM:

```bash
export GEMINI_MASTER_IP=127.0.0.1
export GEMINI_MASTER_PORT=29501
```

For the separate two-node Gemini smoke test, use
`artifact_evaluation/test_gemini_multinode.sh`.

## Figure 8 Tiny

```bash
cd artifact_evaluation/test_evaluation/evaluation/throughput
bash get_throughput_single_node.sh
```

This runs one small microbenchmark per baseline and generates:

* `fig8_tiny.csv`
* `fig8_tiny.png`

If you already generated Figure 8 before Gemini was added, rerun:

```bash
bash get_throughput_single_node.sh --force
```

## Figure 9 Tiny

Run this after Figure 8 Tiny, because it reuses `fig8_tiny.csv`.

```bash
cd artifact_evaluation/test_evaluation/evaluation/throughput
bash get_goodput.sh
```

This generates:

* `fig9_tiny.csv`
* `fig9_tiny.png`

## Figure 11 Tiny

```bash
cd artifact_evaluation/test_evaluation/evaluation/sensitivity_analysis
python run_microbenchmarks.py
```

This generates:

* `fig11_tiny.csv`
* `fig11_tiny.png`

If you already generated Figure 11 before Gemini was added, rerun:

```bash
python run_microbenchmarks.py --force
```

## Figure 12 Tiny

```bash
cd artifact_evaluation/test_evaluation/evaluation/sensitivity_analysis
python run_pccheck_async.py
```

This generates:

* `fig12_tiny.csv`
* `fig12_tiny.png`

## Faster Or Smaller Runs

All scripts accept small-scale knobs. For example:

```bash
python run_microbenchmarks.py --sizes 1 4 --iterations 5
python run_pccheck_async.py --size-mb 4 --iterations 5 --cfreqs 0 1 5 10
bash get_throughput_single_node.sh --size-mb 4 --iterations 5
python run_microbenchmarks.py --baselines gemini --sizes 10 --iterations 5 --force
```

The microbenchmark warmup is 3 iterations, so keep `--iterations` greater than
3.
