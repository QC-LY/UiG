#!/bin/bash
#SBATCH -N {your_node_number}
#SBATCH -p {your_partition}
#SBATCH --gres=gpu:{your_gpu_number}
#SBATCH --job-name=UiG
#SBATCH -o ./logs/infer_slurm_%j.out
#SBATCH --cpus-per-task={your_cpu_number}

module load cuda/12.4 compilers/gcc-11.1.0 compilers/icc-2023.1.0 cmake/3.27.0
export CXX=$(which g++)
export CC=$(which gcc)
export CPLUS_INCLUDE_PATH={your_cuda_include_path}:$CPLUS_INCLUDE_PATH
export CUDA_HOME={your_cuda_home}

export PATH="{your_anaconda_path}:$PATH"
source activate
conda activate UiG

python infer.py \
    --prompt_file ./prompts/test_prompt.txt \
    --log_file ./logs/test_infer.log \
    --output_dir ./outputs \
    --ckpt_path ./ckpts \
    --save_intermediate