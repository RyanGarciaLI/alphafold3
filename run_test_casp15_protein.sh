#!/bin/bash
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_enable_triton_gemm=false"
#export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter" 
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95
export JAX_TRACEBACK_FILTERING=off
PYTHON_BIN="/usr/local/envs/af3/bin/python"

$PYTHON_BIN run_alphafold.py \
    --json_path casp15_protein_msa/7qih_a_b_data_seed1.json \
    --output_dir output_casp15_protein \
    --model_dir /content/ \
    --flash_attention_implementation=xla \
    --norun_data_pipeline