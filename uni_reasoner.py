import time
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import torch
import yaml
import os

class UiGReasoner:
    
    def __init__(self, inferencer, logger):
        """
        Initialize the pipeline with a Bagel inferencer
        
        Args:
            inferencer: InterleaveInferencer instance
        """
        self.inferencer = inferencer
        self.logger = logger
        
        # Default hyperparameters for different tasks
        self.understanding_hyper = {
            'max_think_token_n': 1000,
            'do_sample': False,
            # 'text_temperature': 0.3,
        }
        
        self.generation_hyper = {
            # 'max_think_token_n': 1000,
            # 'do_sample': False,
            # 'text_temperature': 0.3,
            'cfg_text_scale': 4.0,
            'cfg_img_scale': 1.0,
            'cfg_interval': [0.4, 1.0],
            'timestep_shift': 3.0,
            'num_timesteps': 50,
            'cfg_renorm_min': 0.0,
            'cfg_renorm_type': "global",
        }

        self.generation_hyper_think = {
            'max_think_token_n': 1000,
            'do_sample': False,
            # 'text_temperature': 0.3,
            'cfg_text_scale': 4.0,
            'cfg_img_scale': 1.0,
            'cfg_interval': [0.4, 1.0],
            'timestep_shift': 3.0,
            'num_timesteps': 50,
            'cfg_renorm_min': 0.0,
            'cfg_renorm_type': "global",
        }
        
        self.editing_hyper = {
            'max_think_token_n': 1000,
            'do_sample': False,
            # 'text_temperature': 0.3,
            'cfg_text_scale': 4.0,
            'cfg_img_scale': 2.0,
            'cfg_interval': [0.0, 1.0],
            'timestep_shift': 3.0,
            'num_timesteps': 50,
            'cfg_renorm_min': 0.0,
            'cfg_renorm_type': "text_channel",
        }
    
    def decompose_prompt(self, user_prompt: str, prompt_dir: str) -> str:
        """decompose user prompt into detailed steps"""
        with open(os.path.join(prompt_dir, 'decomposition_prompt.txt'), 'r', encoding='utf-8') as f:
            decomposition_prompt = f.read().format(user_prompt=user_prompt)

        self.logger.info("🔍 Decomposing prompt...")
        output_dict = self.inferencer(
            text=decomposition_prompt,
            understanding_output=True,
            think=True,
            **self.understanding_hyper
        )
        
        detailed_prompt = output_dict['text']
        return detailed_prompt
    
    def generate_initial_image(self, detailed_prompt: str, think: bool = False) -> Image.Image:
        """generate initial image using detailed prompt"""
        self.logger.info("🎨 Generating initial image...")
        if think:
            output_dict = self.inferencer(
                text=detailed_prompt,
                # understanding_output=False,
                think=True,
                **self.generation_hyper_think
            )
        else:
            output_dict = self.inferencer(
                text=detailed_prompt,
                # understanding_output=False,
                **self.generation_hyper
            )
        
        image = output_dict['image']
        self.logger.info("✅ Initial image generated successfully")
        return image
    
    def evaluate_image(self, image: Image.Image, original_prompt: str, prompt_dir: str = "./prompts") -> Tuple[bool, str]:
        """
        Evaluate if generated image matches original prompt
        
        Args:
            image: Generated image from previous step
            original_prompt: Original user prompt
            
        Returns:
            Tuple of (needs_editing: bool, editing_instructions: str)
        """

        with open(os.path.join(prompt_dir, 'evaluation_prompt.txt'), 'r', encoding='utf-8') as f:
            evaluation_prompt = f.read().format(original_prompt=original_prompt)

        self.logger.info("🔍 Evaluating image against original prompt...")
        output_dict = self.inferencer(
            image=image,
            text=evaluation_prompt,
            understanding_output=True,
            think=True,
            **self.understanding_hyper
        )
        
        evaluation_result = output_dict['text']
        # self.logger.info(f"📋 Evaluation result: {evaluation_result[:100]}...")
        
        if "MATCH:" in evaluation_result:
            return False, ""
        elif "EDIT_NEEDED:" in evaluation_result:
            editing_instructions = evaluation_result.split("EDIT_NEEDED:")[1].strip()
            return True, editing_instructions
        else:
            # Fallback: assume editing needed if format not followed
            return True, evaluation_result
    
    def edit_image(self, image: Image.Image, editing_instructions: str) -> Image.Image:
        """
        Edit image based on evaluation feedback
        
        Args:
            image: Original generated image
            editing_instructions: Specific editing instructions from previous step
            
        Returns:
            Edited PIL Image
        """
        self.logger.info("✏️ Editing image based on feedback...")
        output_dict = self.inferencer(
            image=image,
            text=editing_instructions,
            # understanding_output=False,
            think=True,
            **self.editing_hyper
        )
        
        edited_image = output_dict['image']
        self.logger.info("✅ Image editing completed")
        return edited_image
    
    def generate_image_with_pipeline(
        self, 
        user_prompt: str, 
        max_iterations: int = 2,
        save_intermediate: bool = False,
        decompose_prompt: bool = True,
        think: bool = False,
        output_dir: str = "./pipeline_outputs",
        prompt_dir: str = "./prompts"
    ) -> Dict[str, Any]:
        """
        Complete pipeline: decompose -> generate -> evaluate -> edit
        
        Args:
            user_prompt: Original user prompt
            max_iterations: Maximum number of editing iterations
            save_intermediate: Whether to save intermediate results
            output_dir: Directory to save intermediate results
            prompt_dir: Directory to save prompts
            decompose_prompt: Whether to decompose prompt
            think: Whether to use thinking for editing
            
        Returns:
            Dictionary containing all pipeline results
        """
        self.logger.info(f"🚀 Starting advanced image generation pipeline...")
        self.logger.info(f"📝 Original prompt: {user_prompt}")
        self.logger.info("=" * 60)
        
        results = {
            'original_prompt': user_prompt,
            'detailed_prompt': '',
            'initial_image': None,
            'final_image': None,
            'evaluation_history': [],
            'editing_history': [],
            'iterations': 0
        }
        
        try:
            if decompose_prompt:
                # Decompose prompt
                detailed_prompt = self.decompose_prompt(user_prompt, prompt_dir)
                results['detailed_prompt'] = detailed_prompt
            else:
                detailed_prompt = user_prompt
            
            # Generate initial image
            current_image = self.generate_initial_image(detailed_prompt, think)
            results['initial_image'] = current_image
            
            if save_intermediate:
                import os
                os.makedirs(output_dir, exist_ok=True)
                current_image.save(f"{output_dir}/initial_image.png")
            
            # Iterative evaluation and editing
            for iteration in range(max_iterations):
                self.logger.info(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
                
                # Evaluate current image
                needs_editing, editing_instructions = self.evaluate_image(
                    current_image, user_prompt, prompt_dir
                )
                
                results['evaluation_history'].append({
                    'iteration': iteration + 1,
                    'needs_editing': needs_editing,
                    'instructions': editing_instructions
                })
                
                if not needs_editing:
                    self.logger.info("✅ Image matches prompt satisfactorily!")
                    break
                
                # Edit image if needed
                if editing_instructions:
                    if save_intermediate:
                        with open(os.path.join(output_dir, f'editing_instructions.txt'), 'a', encoding='utf-8') as f:
                            f.write(editing_instructions + '\n\n')
                    edited_image = self.edit_image(current_image, editing_instructions)
                    
                    results['editing_history'].append({
                        'iteration': iteration + 1,
                        'instructions': editing_instructions,
                        'image': edited_image
                    })
                    
                    current_image = edited_image
                    
                    if save_intermediate:
                        current_image.save(f"{output_dir}/edited_image_iter_{iteration + 1}.png")
                
                results['iterations'] = iteration + 1
            
            results['final_image'] = current_image
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("🎉 Pipeline completed successfully!")
            self.logger.info(f"📊 Total iterations: {results['iterations']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            raise e
    
    def update_hyperparameters(
        self, 
        understanding_hyper: Optional[Dict] = None,
        generation_hyper: Optional[Dict] = None,
        editing_hyper: Optional[Dict] = None
    ):
        """
        Update hyperparameters for different pipeline stages
        
        Args:
            understanding_hyper: Hyperparameters for understanding tasks
            generation_hyper: Hyperparameters for generation tasks  
            editing_hyper: Hyperparameters for editing tasks
        """
        if understanding_hyper:
            self.understanding_hyper.update(understanding_hyper)
        if generation_hyper:
            self.generation_hyper.update(generation_hyper)
        if editing_hyper:
            self.editing_hyper.update(editing_hyper)
        
        self.logger.info("Hyperparameters updated successfully")
