# PCcheck GPU container and two-node GCP runbook

This image packages PCcheck, its native checkpoint libraries, DeepSpeed 0.12.6,
and the pinned Transformers source tree used by the distributed LLM scripts. It
targets the Tesla T4 compute capability (`sm_75`) by default.

## Build and test locally

Run from the repository root:

```bash
docker build -f docker/Dockerfile -t pccheck:cuda12.1 .
docker run --rm --gpus all pccheck:cuda12.1 \
  python -c 'import torch, deepspeed, checkpoint_eval; print(torch.cuda.get_device_name(0), deepspeed.__version__)'
```

The Docker build context must be the repository root, not `docker/`. To target a
different NVIDIA architecture, pass `--build-arg CUDA_ARCH=sm_80` (A100), for
example.

## Push to Artifact Registry

Set these for your project and chosen Artifact Registry region:

```bash
export PROJECT_ID=your-project-id
export REGION=us-central1
export REPOSITORY=distributed-training
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/pccheck:cuda12.1"

gcloud services enable artifactregistry.googleapis.com compute.googleapis.com \
  --project "${PROJECT_ID}"
gcloud artifacts repositories describe "${REPOSITORY}" \
  --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPOSITORY}" \
  --repository-format docker --location "${REGION}" --project "${PROJECT_ID}"
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
docker tag pccheck:cuda12.1 "${IMAGE}"
docker push "${IMAGE}"
```

For a remote build that does not need local disk space, run from the repository
root. Cloud Build pushes the result named by `_IMAGE` automatically:

```bash
gcloud builds submit . --project "${PROJECT_ID}" \
  --config docker/cloudbuild.yaml \
  --substitutions "_IMAGE=${IMAGE}"
```

## Create the two T4 VMs

The following creates two standard (non-Spot) `n1-standard-4` instances: four
vCPUs, 15 GB RAM, and one T4 each. Both receive the same network tag. Resolve
the current Deep Learning VM image to a fixed image name so both nodes boot an
identical OS and NVIDIA driver:

```bash
export ZONE=us-central1-a
export NETWORK=default
export NETWORK_TAG=distributed-training
export VM1=distributed-gpu-1
export VM2=distributed-gpu-2
export VM_SERVICE_ACCOUNT="distributed-training-vm@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud compute accelerator-types describe nvidia-tesla-t4 \
  --zone "${ZONE}" --project "${PROJECT_ID}"

gcloud iam service-accounts describe "${VM_SERVICE_ACCOUNT}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud iam service-accounts create distributed-training-vm \
  --display-name 'Distributed training VMs' --project "${PROJECT_ID}"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${VM_SERVICE_ACCOUNT}" \
  --role roles/artifactregistry.reader

gcloud compute firewall-rules describe distributed-training-internal \
  --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud compute firewall-rules create distributed-training-internal \
  --network "${NETWORK}" --direction INGRESS --action ALLOW \
  --source-tags "${NETWORK_TAG}" --target-tags "${NETWORK_TAG}" \
  --rules tcp,udp,icmp \
  --project "${PROJECT_ID}"

export DLVM_IMAGE="$(gcloud compute images describe-from-family \
  common-cu129-ubuntu-2204-nvidia-580 \
  --project deeplearning-platform-release --format='value(selfLink)')"

gcloud compute instances create "${VM1}" "${VM2}" \
  --project "${PROJECT_ID}" --zone "${ZONE}" \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --maintenance-policy TERMINATE --restart-on-failure \
  --boot-disk-size 100GB --boot-disk-type pd-balanced \
  --image "${DLVM_IMAGE}" \
  --service-account "${VM_SERVICE_ACCOUNT}" --scopes cloud-platform \
  --network "${NETWORK}" --tags "${NETWORK_TAG}"
```

After boot, use `gcloud compute ssh` to install/configure Docker if the selected
DLVM release does not already expose `docker` and the NVIDIA runtime. Generate
a dedicated cluster SSH key, put its public key in the VM user's
`~/.ssh/authorized_keys` and in `$HOME/distributed-ssh/authorized_keys` on both
VMs, and put the private key at `$HOME/distributed-ssh/id_ed25519` on both.
Verify both directions using the internal IPs before launching containers:

```bash
gcloud compute instances list --filter="name=(${VM1} ${VM2})" \
  --zones "${ZONE}" --format='table(name,networkInterfaces[0].networkIP)'
ssh USER@VM1_INTERNAL_IP hostname
ssh USER@VM2_INTERNAL_IP hostname
```

## Run on both GPU VMs

The VMs need an NVIDIA driver, Docker with NVIDIA Container Toolkit, permission
to read Artifact Registry, and tag-restricted internal TCP/UDP access between
the workers. NCCL uses dynamic peer ports after rendezvous. VM-to-VM SSH on port
22 is separate from the container SSH endpoint on port 2222.

Put the same dedicated cluster SSH key and `authorized_keys` in
`$HOME/distributed-ssh` on both VMs. On each VM:

```bash
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker pull "${IMAGE}"
docker run -d --name pccheck --restart unless-stopped \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e START_SSHD=1 \
  -e PDSH_RCMD_TYPE=ssh \
  -e 'PDSH_SSH_ARGS_APPEND=-p 2222 -o StrictHostKeyChecking=no' \
  -e SSH_PORT=2222 \
  -e HF_HOME=/models/huggingface \
  -e OUTPUT_DIR=/checkpoints/pccheck \
  -v "$HOME/distributed-ssh:/ssh-host:ro" \
  -v pccheck-models:/models \
  -v pccheck-checkpoints:/checkpoints \
  "${IMAGE}" sleep infinity
docker exec pccheck nvidia-smi
```

The launcher detects the default private network interface for NCCL. Set
`NCCL_SOCKET_IFNAME` explicitly only when the VM has multiple candidate
interfaces and the default route is not the training network.

Suppose the internal addresses are `10.128.0.2` and `10.128.0.3`. Create the
same hostfile inside both containers:

```bash
printf 'root@10.128.0.2 slots=1\nroot@10.128.0.3 slots=1\n' > /tmp/hostfile
docker cp /tmp/hostfile pccheck:/workspace/pccheck/hostfile
```

Check container-to-container SSH from each VM:

```bash
docker exec pccheck ssh -p 2222 root@10.128.0.2 hostname
docker exec pccheck ssh -p 2222 root@10.128.0.3 hostname
```

Start the bundled two-node smoke test from VM 1:

```bash
docker exec -it pccheck bash -lc \
  'cd /workspace/pccheck && HOSTFILE=/workspace/pccheck/hostfile MASTER_ADDR=10.128.0.2 bash checkpoint_eval/models/llm_distr/run.sh pccheck'
```

The container image contains the PCcheck, DeepSpeed, and Transformers code, but
does not contain model weights or generated checkpoints. The first run downloads
the configured tokenizer to the external `pccheck-models` volume. Training output is
written to the external `pccheck-checkpoints` volume. PCcheck creates one
rank-specific sparse mmap file there; its apparent size can be much larger than
the disk blocks actually allocated. Replace the named volumes
with persistent-disk or shared-filesystem bind mounts when the data must outlive
a VM, and adjust the script environment documented by
`checkpoint_eval/models/llm_distr/run.sh --help`.

## Operational notes

- A T4 has 16 GB of VRAM. Start with the included tiny configuration; the
  paper's OPT-2.7B/A100 settings are not expected to fit unchanged.
- `--network host` avoids Docker NAT issues for NCCL and DeepSpeed rendezvous.
- NCCL uses dynamic TCP ports. Allow all internal TCP/UDP only between the
  cluster's source and target network tags; do not expose those ports publicly.
- Stop or delete GPU VMs when idle to avoid ongoing charges.
