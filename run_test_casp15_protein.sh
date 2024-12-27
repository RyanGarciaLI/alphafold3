#!/bin/bash

PYTHON_BIN="python"

$PYTHON_BIN run_alphafold.py \
    --json_path casp15_protein_msa/7qih_a_b_data.json \
    --output_dir output_casp15_protein \
    --norun_data_pipeline