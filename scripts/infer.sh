export PATH="{your_anaconda_path}:$PATH"
source activate
conda activate UiG

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --prompt_file ./prompts/test_prompt.txt \
    --log_file ./logs/test_infer.log \
    --output_dir ./outputs \
    --ckpt_path ./ckpts \
    --save_intermediate