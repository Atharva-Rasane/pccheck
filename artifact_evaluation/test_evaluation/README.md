# Tiny Artifact Evaluation

This directory mirrors the full artifact-evaluation workflow, but it uses only
the microbenchmarks under `checkpoint_eval/models/microbenchmarks` and writes
small PNG/CSV outputs. It is intended as a quick smoke test for generating the
same four figure shapes at a much smaller scale.

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

## Figure 8 Tiny

```bash
cd artifact_evaluation/test_evaluation/evaluation/throughput
bash get_throughput_single_node.sh
```

This runs one small microbenchmark per baseline and generates:

* `fig8_tiny.csv`
* `fig8_tiny.png`

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
```

The microbenchmark warmup is 3 iterations, so keep `--iterations` greater than
3.
