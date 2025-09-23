import os
import json
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import logging

from PIL import Image
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
from inferencer import InterleaveInferencer
from uni_reasoner import UiGReasoner

# setup logging
def setup_logging(log_file=None):
    """setup logging"""
    if log_file is None:
        log_file = "uni_infer.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_inferencer(ckpt_path):
    model_path = ckpt_path  
    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    max_mem_per_gpu = "80GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    logger.info(f"Device map: {device_map}")

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )

    model = model.eval()
    logger.info('Model loaded')

    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )

    return inferencer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_iterations', type=int, default=4, help="Maximum number of iterations.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--ckpt_path', type=str, default="./ckpts", help="Path to the checkpoint.")
    parser.add_argument('--prompt_file', type=str, default='', help="Path to the prompt file.")
    parser.add_argument('--prompt_text', type=str, default='', help="Prompt text.")
    parser.add_argument('--log_file', type=str, default='./logs/infer.log', help="Path to the log file.")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Path to the output directory.")
    parser.add_argument('--save_intermediate', action='store_true', default=False, help="Save intermediate images.")
    
    # parse arguments
    args = parser.parse_args()
    
    # setup logging
    logger = setup_logging(args.log_file)

    set_seed(args.seed)

    inferencer = init_inferencer(args.ckpt_path)
    pipeline = UiGReasoner(inferencer, logger)

    if os.path.exists(args.prompt_file):
        logger.info(f"Processing prompt file: {args.prompt_file}")
        with open(args.prompt_file, 'r') as f:
            prompts = f.readlines()

        for idx, prompt in tqdm(enumerate(prompts)):
            prompt = prompt.strip()
            output_dir = os.path.join(args.output_dir, prompt.replace(" ", "_"))
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
                f.write(prompt)

            results = pipeline.generate_image_with_pipeline(
                user_prompt=prompt,
                max_iterations=args.max_iterations,
                save_intermediate=args.save_intermediate,
                decompose_prompt=False,
                output_dir=output_dir
            )

            results['final_image'].save(os.path.join(output_dir, "final_image.png"))
    else:
        if args.prompt_text:
            output_dir = os.path.join(args.output_dir, args.prompt_text.replace(" ", "_"))
            os.makedirs(output_dir, exist_ok=True)

            results = pipeline.generate_image_with_pipeline(
                user_prompt=args.prompt_text,
                max_iterations=args.max_iterations,
                save_intermediate=args.save_intermediate,
                decompose_prompt=False,
                output_dir=output_dir
            )

            results['final_image'].save(os.path.join(output_dir, "final_image.png"))
        else:
            raise ValueError("Either prompt file or prompt text must be provided.")