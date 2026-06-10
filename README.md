# PCcheck

PCcheck is a concurrent checkpoint mechanism for ML training. It is based on our ASPLOS'25 paper: "PCcheck: Persistent Concurrent Checkpointing for ML".

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Hardware Requirement](#hardware-requirement)
- [Hardware Configuration used in the paper](#hardware-configuration-used-in-the-paper)
- [Installation](#installation)
- [Example](#example)
- [ASPLOS'25 Artifact evaluation](#asplos25-artifact-evaluation)
- [Paper](#paper)

## Introduction

PCcheck is mechanism that allows frequent checkpointing for ML training workloads. The key idea behind PCcheck is to allow multiple concurrent checkpoints in parallel, thus allowing training to make progress and not stalling waiting for checkpoints to persist.

PCcheck optimizes copying and persisting a checkpoint by employing chunking and pipelining GPU-CPU copies and CPU-storage copies, and multiple threads for persisting to storage.

PCcheck optimizes the number of in-flight concurrent checkpoints, chunk size, and number of threads based on the training workload and system characteristics (e.g. storage bandwidth).

## Project Structure

The repo is structured as follows:

```
> tree .
├── checkpoint_eval
|   ├── checkfreq                  # Integration code for CheckFreq
|   ├── deepspeed                  # Necessary modifications for Deepseed
|   ├── gemini                     # Our implementation of Gemini
|   ├── gpm                        # Integration code for GPM
|   ├── models                     # Code and scripts for the models used in our evaluation
|   ├── pccheck                    # PCcheck implementation
├── artifact_evaluation            # Scripts and instructions for the ASPLOS'25 Artifact Evaluation
|   ├── evaluation                 # Scripts for reproducing key figures from the paper's evaluation sect
|   |   ├── sensitivity analysis   # Scripts for Figures 11, 12
|   |   ├── throughput             # Scripts for Figures 8, 9

```

## Hardware Configurations used in the paper

We used a2-highgpu-1g VMs from Google Cloud Platform. Each VM has an A100-40GB GPU attached, 1TB
of pd-ssd, 12 vCPUs, and 85 GB of DRAM.

## Installation

Preinstallitions for newly created VM
```
# Update Ubuntu's package list so apt knows the latest available package versions
sudo apt-get update

# Install basic tools needed before pulling the repo and running its installer:
# - git: clone/pull the GitHub repository
# - build-essential: compiler tools like gcc/g++/make, needed for packages that compile code
# - wget/curl: download files/scripts from the internet
# - ca-certificates: lets HTTPS downloads verify certificates correctly
# - software-properties-common: provides add-apt-repository
# - gnupg: handles repository signing keys
# - lsb-release: helps scripts detect your Ubuntu version/codename
sudo apt-get install -y \
  git \
  build-essential \
  wget \
  curl \
  ca-certificates \
  software-properties-common \
  gnupg \
  lsb-release

# Add the Deadsnakes PPA, which provides Python versions not always available
# in Ubuntu's default repositories, including Python 3.9 on some systems
sudo add-apt-repository -y ppa:deadsnakes/ppa

# Update package list again so apt includes packages from the new Python PPA
sudo apt-get update

# Install Python 3.9 and related development components:
# - python3.9: the Python interpreter
# - python3.9-dev: headers needed to compile Python packages with native extensions
# - python3.9-distutils: packaging/build helper used by some Python installers
# - python3.9-venv: lets you create Python 3.9 virtual environments
sudo apt-get install -y \
  python3.9 \
  python3.9-dev \
  python3.9-distutils \
  python3.9-venv

# Download the official pip bootstrap installer
# pip is Python's package installer; the repo's install scripts will need it
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip specifically for Python 3.9
python3.9 get-pip.py

# Clone the PCCHECK repository from GitHub to your current directory
git clone https://github.com/Atharva-Rasane/pccheck.git

# Move into the downloaded repository folder
cd pccheck

# Make the repo's installer scripts executable
# This lets you run them directly with ./script-name.sh
chmod +x install_preq_at_vm.sh install.sh

# Run the repo's prerequisite installer
# This is expected to install heavier dependencies such as CUDA, Docker,
# NVIDIA container tools, system libraries, and other VM-level requirements
./install_preq_at_vm.sh

# Run the main project installer
# This should install the Python/project dependencies needed by PCCHECK
./install.sh
```

We used VMs with Ubuntu 20.04 (ubuntu-2004-focal-v20240830 from GCP).
We then used the [install_preq_at_vm.sh](install_preq_at_vm.sh) to install all required packages.
Finally, we run the [install.sh](install.sh) to build and install PCcheck and the rest baselines.

## Example

After installing, you can run [test_simple.sh](artifact_evaluation/test_simple.sh) to check everything is in place. This script trains a VGG16 model checkpointing every 50 iterations.

## ASPLOS'25 Artifact evaluation

We provide instructions for evaluating key results from our paper under the [artifact_evaluation](artifact_evaluation) directory.

## Paper

If you use PCcheck, please cite our paper:
```bibtex
@inproceedings {asplos25pccheck,
  author = {Strati Foteini and Friedman Michal and Klimovic Ana},
  title = {PCcheck: Persistent Concurrent Checkpointing for ML},
  booktitle = {},
  year = {2025},
  isbn = {},
  address = {},
  pages = {},
  url = {},
  doi = {},
  publisher = {Association for Computing Machinery},
}
```