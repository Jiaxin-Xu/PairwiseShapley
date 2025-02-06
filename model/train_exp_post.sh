#!/bin/bash

# Ensure conda environment initialization (may be a no-op if already initialized)
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate myenv
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment 'myenv'. Exiting."
  exit 1
fi

# Ensure model version is provided as a command line argument
if [ -z "$1" ]; then
  echo "Model version not specified. Usage: $0 <model_version>"
  exit 1
fi

# Define model version from command line argument
MODEL_VERSION=$1

# Log file to capture output
LOGFILE="explain_batch_all_log_base_model_${MODEL_VERSION}.txt"
echo "Logging to $LOGFILE"

# First, run the training script
python train.py --model_version $MODEL_VERSION >> "$LOGFILE" 2>&1
if [ $? -ne 0 ]; then
  echo "Error running train.py. Check $LOGFILE for details."
  exit 1
else
  echo "Training completed." >> "$LOGFILE"
  echo "Training completed."
fi

# Define an array of methods
methods=("Baseline-0" "Baseline-median" "Pairwise" "Pairwise-random" "Pairwise-comps" "Pairwise-sim" "Marginal-all" "Marginal-kmeans" "Uniform" "TreeShap-treepath" "Conditional-all")

# Function to run python script with error handling
run_python() {
  method=$1
  python explain.py --model_version $MODEL_VERSION --num_pairs 50 --explain_method "$method" >> "$LOGFILE" 2>&1
  if [ $? -ne 0 ]; then
    echo "Error running explain.py with --explain_method $method. Check $LOGFILE for details."
  else
    echo "Done with $method." >> "$LOGFILE"
    echo "Done with $method."
  fi
}

# Sequential execution
for method in "${methods[@]}"; do
  run_python "$method"
done

# # Finally, run the post-processing script
# python post_exp.py --model_version $MODEL_VERSION >> "$LOGFILE" 2>&1
# if [ $? -ne 0 ]; then
#   echo "Error running post_exp.py. Check $LOGFILE for details."
#   exit 1
# else
#   echo "Post-processing completed." >> "$LOGFILE"
#   echo "Post-processing completed."
# fi

echo "All tasks completed."