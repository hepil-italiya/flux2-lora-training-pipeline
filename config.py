import torch

class Config:
    repo_id = "diffusers/FLUX.2-dev-bnb-4bit"  # Quantized text-encoder and DiT. VAE still in bf16
    device = "cuda:0"
    torch_dtype = torch.bfloat16

    DATA_FOLDER = "Train Folder Path"
    
    #MODEL SAVE 
    SAVE_PATH = "Model Save Path"
    MODEL_SAVE_INTERATION = 2000
    EPOCHS = 100
    
    # HYPERPARAMETERS
    LEARNING_RATE = 5e-5 #7.5e-5 #1e-4
    LORA_RANK = 64
    LORA_ALPHA = 32
    BATCH_SIZE = 1 
    WEIGHT_DECAY = 0.05
    TARGET_MODULES = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2"]
            
    GUIDANCE_SCALE = 1.0  # Typically 1.0 for training (no CFG)
    NUM_IMAGES_PER_PROMPT = 1  # Fixed for now
    
    IMAGE_HEIGHT = 1024
    IMAGE_WEIGHT = 1024
    
    # TEXT EMBEDDING
    MAX_SEQUENCE_LENGTH = 512

config = Config()