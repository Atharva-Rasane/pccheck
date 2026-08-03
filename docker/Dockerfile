FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCH=sm_75
ARG TRANSFORMERS_COMMIT=ee88ae59940fd4b2c8fc119373143d7a1175c651

ENV CUDA_HOME=/usr/local/cuda-12.1 \
    DEEPSPEED_DISABLE_MPI=1 \
    HF_HOME=/data/huggingface \
    NCCL_DEBUG=WARN \
    PCCHECK_HOME=/workspace/pccheck \
    PYTHON=python \
    PYTHONUNBUFFERED=1 \
    SSH_PORT=2222 \
    TRANSFORMERS_DIR=/workspace/transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libatomic1 \
        libpci-dev \
        libpmem-dev \
        ninja-build \
        openssh-client \
        openssh-server \
        pdsh \
        rsync \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /data/huggingface /workspace

WORKDIR /workspace/pccheck
COPY . .

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir "datasets==2.14.7" "pyarrow<21" \
    && python -m pip install --no-cache-dir -e . --config-settings editable_mode=compat \
    && git clone https://github.com/huggingface/transformers.git "${TRANSFORMERS_DIR}" \
    && git -C "${TRANSFORMERS_DIR}" checkout "${TRANSFORMERS_COMMIT}" \
    && python -m pip install --no-cache-dir -e "${TRANSFORMERS_DIR}" \
    && make -C checkpoint_eval/gpm clean \
    && make -C checkpoint_eval/gpm CUDA_ARCH="${CUDA_ARCH}" \
    && make -C checkpoint_eval/pccheck clean \
    && make -C checkpoint_eval/pccheck libtest_ssd.so \
    && bash checkpoint_eval/models/llm_distr/run.sh --copy-only \
    && rm -rf /root/.cache/pip "${TRANSFORMERS_DIR}/.git"

COPY docker/entrypoint.sh /usr/local/bin/pccheck-entrypoint
RUN chmod 0755 /usr/local/bin/pccheck-entrypoint \
    && printf '%s\n' \
        'Port 2222' \
        'PermitRootLogin prohibit-password' \
        'PasswordAuthentication no' \
        'PubkeyAuthentication yes' \
        >> /etc/ssh/sshd_config \
    && passwd -d root \
    && rm -f /etc/ssh/ssh_host_*

EXPOSE 2222 1234 1235 29500-29600
ENTRYPOINT ["/usr/local/bin/pccheck-entrypoint"]
CMD ["bash"]
