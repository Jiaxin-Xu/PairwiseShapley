#!/bin/bash

# Ensure conda environment initialization (may be a no-op if already initialized)
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate myenv
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment 'myenv'. Exiting."
  exit 1
fi

# Datasets and model versions
datasets=("kingcounty" "hiv" "polymer")

# Function to run python script with error handling
run_python() {
  dataset=$1
  model_version=$2
  method=$3
  LOGFILE="explain_single50_log_${dataset}_model_${model_version}.txt"

  # Create log file if it doesn't exist
  if [ ! -f "$LOGFILE" ]; then
    touch "$LOGFILE"
  fi

  # Check if the combination is already done
  if grep -q "Done with ${method}" "$LOGFILE"; then
    echo "Skipping ${dataset} - ${model_version} - ${method}, already done."
    return
  fi
  
  python explain.py --dataset "$dataset" --model_version "$model_version" --num_pairs 1 --explain_method "$method" >> "$LOGFILE" 2>&1
  if [ $? -ne 0 ]; then
    echo "Error running explain.py with --explain_method $method. Check $LOGFILE for details."
  else
    echo "Done with $method." >> "$LOGFILE"
    echo "Done with $method."
  fi
}

# Loop through each dataset
for dataset in "${datasets[@]}"; do
  if [ "$dataset" = "kingcounty" ]; then
    model_versions=("v3" "v2" "v1" "v0")
    methods=("Baseline-0" "Baseline-median" "Pairwise-random" "Pairwise-comps" "Pairwise-sim" "Marginal-all" "Marginal-kmeans" "Uniform" "TreeShap-treepath" "Conditional-all")
  else
    model_versions=("v0")
    methods=("Baseline-0" "Baseline-median" "Pairwise-random" "Pairwise-sim" "Marginal-all" "Marginal-kmeans" "Uniform" "TreeShap-treepath")
  fi
  
  # Loop through each model version
  for model_version in "${model_versions[@]}"; do
    # Sequential execution of methods
    for method in "${methods[@]}"; do
      run_python "$dataset" "$model_version" "$method"
    done
  done
done

echo "Running Time: All tasks completed."