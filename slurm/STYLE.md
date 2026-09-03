# Slurm Script Style Guide

This document outlines the conventions and style guidelines for writing Slurm scripts in this repository, based on the existing scripts.

## General Conventions
- **File Naming**: Prefix scripts that execute tasks or allocate resources with `run_` (e.g., `run_latk_binvox_1.sh`, `run_salloc_1.sh`, `run_activate_Example.sh`).
- **Shebang**: Always include `#!/bin/bash` at the very beginning of the script.

## Batch Jobs (`sbatch` scripts)
Batch scripts are used to submit non-interactive jobs to the cluster.

### Standard Headers
Include the following `#SBATCH` directives to request resources and set job parameters:

```bash
#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8      # Adjust based on job needs (e.g., 3, 8)
#SBATCH --mem=30G              # Adjust based on job needs
#SBATCH --time=168:0:00        # Format is hours:minutes:seconds
#SBATCH --account=def-example  # Use your allocated account
#SBATCH --mail-user=example@example.ca
#SBATCH --mail-type=ALL
```

### GPU Allocation
For jobs requiring a GPU, add the `--gres` directive. You can specify the type of GPU if needed:
```bash
#SBATCH --gres=gpu:v100:1      # Requests 1 V100 GPU
```

### Job Execution Structure
1. **Environment Setup**: Activate the required Python virtual environment.
   ```bash
   source ~/Example/bin/activate
   ```
2. **Directory Management**: Change to the appropriate working directory.
   ```bash
   cd /home/$USER/projects/def-example/your_project
   ```
3. **Data Staging (Optional but Recommended)**: Copy data to the compute node's local storage for faster access.
   ```bash
   #cp -r data/ $SLURM_TMPDIR/data
   #cp -r saved_models/ $SLURM_TMPDIR/saved_models
   ```
4. **Execution**: Use `srun` to execute the main script or command.
   ```bash
   srun bash your_script.sh
   ```

## Interactive Jobs (`salloc` scripts)
Interactive scripts allocate resources for real-time development, debugging, or setup tasks.

### Resource Requests
Use the `salloc` command to request resources. Keep different configurations handy as comments for quick swapping:

```bash
#!/bin/bash

# V100-32GB GPU
#salloc --ntasks=1 --gres=gpu:v100:1 --cpus-per-task=8 --mem=30G --constraint=cascade,v100

# Standard V100 GPU
#salloc --ntasks=1 --gres=gpu:v100:1 --cpus-per-task=8 --mem=30G

# Any available GPU
salloc --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=30G

# CPU only
#salloc --ntasks=1 --cpus-per-task=8 --mem=30G
```

### Post-Allocation Steps
After allocating the resources, these scripts typically activate the environment and perform setup tasks (like installing dependencies):
```bash
source ~/Example/bin/activate
cd ~/your-project
pip install --no-index -r requirements.txt
```

## Environment Activation Scripts
For simple scripts that just set up the environment (like `run_activate_Example.sh`), keep it minimal:
```bash
source ~/Example/bin/activate
python --version
```
