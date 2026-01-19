"""
Text Embedding Generator for FLUX 2 LoRA Training Pipeline

This module provides functionality to pre-compute text embeddings for the FLUX 2 
training pipeline. It processes a data folder containing subfolders, each with 
a prompt.txt file, and generates the corresponding prompt embeddings and text IDs.

Usage:
    python text_embedding.py

The script will:
    1. Load the FLUX 2 text encoder pipeline
    2. Iterate through all subfolders in DATA_FOLDER
    3. Read prompt.txt from each subfolder
    4. Generate and save prompt_embeds.pt and text_ids.pt
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
from glob import glob
from tqdm import tqdm

import torch
from diffusers import Flux2Pipeline

from config import config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TextEmbeddingGenerator:
    """
    Generates text embeddings for FLUX 2 training pipeline.
    
    This class loads a minimal FLUX 2 pipeline (text encoder only) and processes
    prompt files to generate pre-computed embeddings for training efficiency.
    
    Attributes:
        repo_id (str): The Hugging Face repository ID for the FLUX 2 model.
        device (str): The device to run the model on (e.g., 'cuda:0').
        torch_dtype (torch.dtype): The data type for model computations.
        max_sequence_length (int): Maximum sequence length for text encoding.
    """
    
    def __init__(
        self,
        repo_id: str = None,
        device: str = None,
        torch_dtype: torch.dtype = None,
        max_sequence_length: int = None
    ):
        """
        Initialize the TextEmbeddingGenerator.
        
        Args:
            repo_id: Hugging Face repository ID. Defaults to config.repo_id.
            device: Device to run on. Defaults to config.device.
            torch_dtype: Data type for computations. Defaults to config.torch_dtype.
            max_sequence_length: Max sequence length. Defaults to config.MAX_SEQUENCE_LENGTH.
        """
        self.repo_id = repo_id or config.repo_id
        self.device = device or config.device
        self.torch_dtype = torch_dtype or config.torch_dtype
        self.max_sequence_length = max_sequence_length or config.MAX_SEQUENCE_LENGTH
        
        self._pipe: Optional[Flux2Pipeline] = None
        
        logger.info(f"TextEmbeddingGenerator initialized with:")
        logger.info(f"  - repo_id: {self.repo_id}")
        logger.info(f"  - device: {self.device}")
        logger.info(f"  - torch_dtype: {self.torch_dtype}")
        logger.info(f"  - max_sequence_length: {self.max_sequence_length}")
    
    def _load_pipeline(self) -> None:
        """
        Lazy-load the FLUX 2 pipeline with only text encoder components.
        
        This loads the pipeline without VAE and transformer to save VRAM,
        as we only need the text encoder for generating embeddings.
        """
        if self._pipe is not None:
            return
        
        logger.info("Loading FLUX 2 pipeline (text encoder only)...")
        
        self._pipe = Flux2Pipeline.from_pretrained(
            self.repo_id,
            scheduler=None,
            vae=None,
            transformer=None,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        logger.info("Pipeline loaded successfully.")
    
    def encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a text prompt into embeddings.
        
        Args:
            prompt: The text prompt to encode.
            
        Returns:
            Tuple containing:
                - prompt_embeds: The encoded prompt embeddings tensor.
                - text_ids: The text position IDs tensor.
                
        Raises:
            ValueError: If prompt is empty or None.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace only.")
        
        # Ensure pipeline is loaded
        self._load_pipeline()
        
        # Encode the prompt
        with torch.no_grad():
            prompt_embeds, text_ids = self._pipe.encode_prompt(
                prompt=prompt.strip(),
                max_sequence_length=self.max_sequence_length
            )
        
        return prompt_embeds, text_ids
    
    def process_folder(
        self,
        folder_path: str,
        prompt_filename: str = "prompt.txt",
        force_regenerate: bool = False
    ) -> bool:
        """
        Process a single folder containing a prompt file.
        
        Args:
            folder_path: Path to the folder containing the prompt file.
            prompt_filename: Name of the prompt file. Defaults to 'prompt.txt'.
            force_regenerate: If True, regenerate even if embeddings exist.
            
        Returns:
            True if embeddings were generated successfully, False otherwise.
        """
        folder_path = Path(folder_path)
        prompt_path = folder_path / prompt_filename
        prompt_embeds_path = folder_path / "prompt_embeds.pt"
        text_ids_path = folder_path / "text_ids.pt"
        
        # Check if prompt file exists
        if not prompt_path.exists():
            logger.warning(f"No prompt file found at: {prompt_path}")
            return False
        
        # Check if embeddings already exist
        if not force_regenerate:
            if prompt_embeds_path.exists() and text_ids_path.exists():
                logger.debug(f"Embeddings already exist for: {folder_path.name}")
                return True
        
        # Read prompt
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read prompt file {prompt_path}: {e}")
            return False
        
        if not prompt:
            logger.warning(f"Empty prompt in: {prompt_path}")
            return False
        
        # Generate embeddings
        try:
            prompt_embeds, text_ids = self.encode_prompt(prompt)
            
            # Save embeddings
            torch.save(prompt_embeds.cpu(), prompt_embeds_path)
            torch.save(text_ids.cpu(), text_ids_path)
            
            logger.debug(f"Generated embeddings for: {folder_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for {folder_path}: {e}")
            return False
    
    def process_data_folder(
        self,
        data_folder: str = None,
        prompt_filename: str = "prompt.txt",
        force_regenerate: bool = False
    ) -> Tuple[int, int]:
        """
        Process all subfolders in the data folder.
        
        Args:
            data_folder: Path to the main data folder. Defaults to config.DATA_FOLDER.
            prompt_filename: Name of the prompt file in each subfolder.
            force_regenerate: If True, regenerate embeddings even if they exist.
            
        Returns:
            Tuple of (successful_count, total_count).
            
        Raises:
            FileNotFoundError: If data_folder doesn't exist.
        """
        data_folder = Path(data_folder or config.DATA_FOLDER)
        
        if not data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
        
        if not data_folder.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {data_folder}")
        
        # Get all subfolders
        subfolders = sorted([
            d for d in data_folder.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if not subfolders:
            logger.warning(f"No subfolders found in: {data_folder}")
            return 0, 0
        
        logger.info(f"Processing {len(subfolders)} subfolders in: {data_folder}")
        
        successful = 0
        
        for folder in tqdm(subfolders, desc="Generating embeddings"):
            if self.process_folder(folder, prompt_filename, force_regenerate):
                successful += 1
        
        logger.info(f"Successfully processed {successful}/{len(subfolders)} folders")
        
        return successful, len(subfolders)
    
    def cleanup(self) -> None:
        """
        Clean up resources and free GPU memory.
        """
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Resources cleaned up.")


def main():
    """
    Main function to generate embeddings for all prompts in the data folder.
    """
    print("\n" + "=" * 60)
    print("FLUX 2 TEXT EMBEDDING GENERATOR")
    print("=" * 60 + "\n")
    
    # Validate data folder path
    data_folder = Path(config.DATA_FOLDER)
    if not data_folder.exists() or str(data_folder) == "Train Folder Path":
        logger.error(
            "Please set a valid DATA_FOLDER path in config.py\n"
            f"Current value: {config.DATA_FOLDER}"
        )
        return
    
    # Initialize generator
    generator = TextEmbeddingGenerator()
    
    try:
        # Process all folders
        successful, total = generator.process_data_folder(
            force_regenerate=False  # Set to True to regenerate all embeddings
        )
        
        print("\n" + "=" * 60)
        print("EMBEDDING GENERATION COMPLETE!")
        print(f"Successfully processed: {successful}/{total} folders")
        print("=" * 60 + "\n")
        
    finally:
        # Clean up
        generator.cleanup()


if __name__ == "__main__":
    main()
