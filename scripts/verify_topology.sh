#!/usr/bin/env bash
set -euo pipefail
nvidia-smi -L
echo
nvidia-smi topo -m
echo
nvidia-smi topo -p2p p || true
echo
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version --format=csv
