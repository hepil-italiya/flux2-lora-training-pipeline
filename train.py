# Python Module
import torch
import torch.nn.functional as F
from diffusers import Flux2Pipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm 
import bitsandbytes as bnb
import os 

# Custom modules
from dataloader import ImageDataset 
from config import config
from save_loss import logger


class Flux2Trainer:
    def __init__(self):
        """
        Initialize FLUX 2 trainer with precomputed embeddings approach
        
        We use precomputed embeddings (due to OOM), which SHOULD contain image features
        from the text encoder\'s vision tower, not just text.
        
        For I2I conditioning, we use BOTH multimodal embeds AND latent concatenation.
        """
        
        print("[INIT] Loading FLUX 2 pipeline (without text encoder to save memory)...")
        # Load without text encoder to save VRAM
        self.__pipe = Flux2Pipeline.from_pretrained(
            config.repo_id, 
            text_encoder=None,  # Skip to save memory
            torch_dtype=config.torch_dtype
        ).to(config.device)
        
        self.__vae = self.__pipe.vae 
        self.__transformer = self.__pipe.transformer

        print("[INIT] Freezing base model weights...")
        # Freeze base model weights
        self.__transformer.requires_grad_(False)
        self.__vae.requires_grad_(False)
        self.__vae.eval()
        
        print("[INIT] Injecting LoRA into transformer...")
        # Inject LoRA
        self.__lora_config = LoraConfig(
            r=config.LORA_RANK,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.TARGET_MODULES,
            bias="none"
        )

        self.__lora_dit = get_peft_model(self.__transformer, self.__lora_config)
        self.__lora_dit.train()
        self.__lora_dit.enable_gradient_checkpointing()
        
        print("\n[LORA] Trainable parameters:")
        self.__lora_dit.print_trainable_parameters()
        
        # Memory optimizations
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cudnn.benchmark = True
        
        print("[INIT] Setting up optimizer...")
        # Optimizer
        self.__optimizer = bnb.optim.Adam8bit(
            self.__lora_dit.parameters(),
            lr=config.LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Create save directory
        if not os.path.exists(config.SAVE_PATH):
            os.makedirs(config.SAVE_PATH)
            print(f"[INIT] Created save directory: {config.SAVE_PATH}")
            
        # Training settings
        self.__batch_size = config.BATCH_SIZE
        self.__device = config.device
        self.__dtype = config.torch_dtype
        self.__min_loss = None
        
        print("[INIT] Initialization complete!\n")

    # ==================== HELPER FUNCTIONS ====================
    
    def __denormalize(self, img_tensor):
        """Denormalize from [-1, 1] to [0, 1]"""
        return (img_tensor / 2 + 0.5).clamp(0, 1)

    def __pack_latents(self, latents):
        """Pack latents: (B, C, H, W) -> (B, H*W, C)"""
        batch_size, num_channels, height, width = latents.shape
        return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

    def __prepare_latent_ids(self, batch_size, height_patched, width_patched, device, dtype=torch.long):
        """
        Prepare 4D position IDs for latents: (T, H, W, L)
        Returns: (B, H*W, 4)
        """
        h_coords = torch.arange(height_patched, device=device, dtype=dtype)
        w_coords = torch.arange(width_patched, device=device, dtype=dtype)
        
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing='ij')
        
        flat_h = grid_h.flatten()
        flat_w = grid_w.flatten()
        seq_len = flat_h.shape[0]
        
        # T=0, L=0 for target latents
        flat_t = torch.zeros(seq_len, device=device, dtype=dtype)
        flat_l = torch.zeros(seq_len, device=device, dtype=dtype)
        
        ids = torch.stack([flat_t, flat_h, flat_w, flat_l], dim=1)
        latent_ids = ids.unsqueeze(0).expand(batch_size, -1, -1)
        
        return latent_ids

    def __prepare_image_ids(self, image_latents_list, batch_size, device, dtype=torch.long, scale=10):
        ids_list = []
        t_offsets = [scale + scale * i for i in range(len(image_latents_list))]
        
        for i, latents in enumerate(image_latents_list):
            # latents: (B, C, H_p, W_p)
            batch_size, _, h_p, w_p = latents.shape
            
            h_coords = torch.arange(h_p, device=device, dtype=dtype)
            w_coords = torch.arange(w_p, device=device, dtype=dtype)
            grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing='ij')
            
            flat_h = grid_h.flatten()
            flat_w = grid_w.flatten()
            seq_len = flat_h.shape[0]
            
            flat_t = torch.full((seq_len,), t_offsets[i], device=device, dtype=dtype)
            flat_l = torch.zeros(seq_len, device=device, dtype=dtype)
            
            ids = torch.stack([flat_t, flat_h, flat_w, flat_l], dim=1)
            ids_list.append(ids)
            
        all_ids = torch.cat(ids_list, dim=0).unsqueeze(0).expand(batch_size, -1, -1)
        return all_ids

    def __patchify_latents(self, latents):
        """
        Patchify latents: (B, 16, H, W) -> (B, 64, H/2, W/2)
        FLUX 2 uses 2x2 patching
        """
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents
    
    # ==================== MODEL SAVING ====================
    
    def __save_min_loss_lora(self, loss):
        """Save LoRA weights when minimum loss is achieved"""
        loss = round(loss, 4)
        min_loss_save_path = os.path.join(config.SAVE_PATH, "Min_Loss_Lora")
        
        if self.__min_loss is None:
            self.__lora_dit.save_pretrained(min_loss_save_path)
            self.__min_loss = loss
            print(f"\n[SAVE] First minimum loss LoRA saved: {loss}")
        else:
            if loss < self.__min_loss:
                self.__lora_dit.save_pretrained(min_loss_save_path)
                old_min = self.__min_loss
                self.__min_loss = loss
                print(f"\n[SAVE] New minimum loss LoRA: {loss} (previous: {old_min})")

    # ==================== TRAINING LOOP ====================
    
    def train(self, dataloader):
        """
        Main training loop with proper I2I conditioning
        
        KEY FEATURES:
        1. Use input_images_tensors for latent conditioning (with spatial concat if multiple)
        2. Precomputed embeddings include image features (from text encoder\'s vision tower)
        3. Concatenate noisy target and source latents in hidden_states
        4. Use mixed position IDs for target and sources
        5. Loss computed ONLY on target predictions
        
        IMPORTANT: Precomputed embeddings should be created using:
        - Text encoder with BOTH text prompt AND conditioning images
        - This creates multimodal embeddings (text + image features)
        """
        
        iteration = 0
        total_steps = len(dataloader) * config.EPOCHS
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.__optimizer, 
            T_max=total_steps, 
            eta_min=1e-7
        )
        
        print(f"[TRAIN] Starting training for {config.EPOCHS} epochs")
        print(f"[TRAIN] Total steps: {total_steps}")
        print(f"[TRAIN] Batch size: {self.__batch_size}\n")

        for epoch in range(config.EPOCHS):
            epoch_loss = 0
            
            pbar = tqdm(
                dataloader, 
                total=len(dataloader), 
                desc=f"Epoch {epoch+1}/{config.EPOCHS}"
            )
            
            for batch_idx, batch in enumerate(pbar):
                self.__optimizer.zero_grad()
                
                # ==================== UNPACK BATCH ====================
                input_images_tensors = batch["input_images_tensors"]  # List of tensors [ (3,H,W), ... ]
                target_img_tensor = batch["output_image_tensor"]      # Tensor (3,H,W)
                
                # Load precomputed embeddings
                # These should be multimodal (text + images from vision tower)
                prompt_embeds = batch["prompt_embeds"].squeeze(0).to(self.__device, self.__dtype)
                text_ids = batch["text_ids"].squeeze(0).to(self.__device, dtype=torch.long)
                
                # ==================== ENCODE TARGET IMAGE ====================
                # Denormalize target for VAE
                target_img_raw = self.__denormalize(target_img_tensor).to(
                    self.__device, self.__dtype
                )#.unsqueeze(0)  # Add batch dim
                
                with torch.no_grad():
                    # VAE encode
                    posterior = self.__vae.encode(target_img_raw)
                    clean_latents = posterior.latent_dist.mode() if hasattr(
                        posterior, "latent_dist"
                    ) else posterior.latent_dist.sample()
                    
                    # Patchify
                    clean_latents = self.__patchify_latents(clean_latents)
                    
                    # Normalize with batch norm stats
                    bn_mean = self.__vae.bn.running_mean.view(1, -1, 1, 1).to(
                        self.__device, self.__dtype
                    )
                    bn_std = torch.sqrt(
                        self.__vae.bn.running_var.view(1, -1, 1, 1) + 
                        self.__vae.config.batch_norm_eps
                    ).to(self.__device, self.__dtype)
                    
                    clean_normalized = (clean_latents - bn_mean) / bn_std
                
                _, _, h_p, w_p = clean_normalized.shape
                
                # ==================== FLOW MATCHING: ADD NOISE ====================
                # Sample random timestep
                num_train_timesteps = 1000
                timesteps = torch.randint(
                    0, num_train_timesteps, 
                    (self.__batch_size,), 
                    device=self.__device
                ).long()
                sigmas = timesteps.float() / num_train_timesteps  # Normalize to [0, 1)
                
                # Flow matching: x_t = (1 - sigma) * x_0 + sigma * noise
                noise = torch.randn_like(clean_normalized)
                noisy_normalized = (
                    (1 - sigmas.view(-1, 1, 1, 1)) * clean_normalized + 
                    sigmas.view(-1, 1, 1, 1) * noise
                )
                
                # Pack noisy latents
                packed_noisy_target = self.__pack_latents(noisy_normalized)
                
                # Prepare position IDs for target
                target_ids = self.__prepare_latent_ids(
                    self.__batch_size, h_p, w_p, self.__device
                )
                
                # ==================== ENCODE SOURCE IMAGES ====================
                # Denormalize sources
                source_imgs_raw = [self.__denormalize(img).to(self.__device, self.__dtype) for img in input_images_tensors]
                
                # If multiple inputs, concatenate horizontally (assume same height)
                if len(source_imgs_raw) > 1:
                    # Concat along width (dim=2 for [3, H, W])
                    concat_img = torch.cat(source_imgs_raw, dim=2)
                    source_imgs_raw = [concat_img]
                
                packed_source_latents_list = []
                source_normalized_list = []  # For IDs
                
                with torch.no_grad():
                    for src_img in source_imgs_raw:
                        # Add batch dim
                        # src_img = src_img.unsqueeze(0)
                        posterior_src = self.__vae.encode(src_img)
                        src_latents = posterior_src.latent_dist.mode() if hasattr(
                            posterior_src, "latent_dist"
                        ) else posterior_src.latent_dist.sample()
                        
                        src_latents = self.__patchify_latents(src_latents)
                        src_normalized = (src_latents - bn_mean) / bn_std
                        source_normalized_list.append(src_normalized)
                        packed_src = self.__pack_latents(src_normalized)
                        packed_source_latents_list.append(packed_src)
                
                # Concat sources if multiple (but after spatial concat, likely 1 list item)
                all_source_latents = torch.cat(packed_source_latents_list, dim=1) if len(packed_source_latents_list) > 1 else packed_source_latents_list[0] if packed_source_latents_list else None
                
                # Prepare source IDs
                source_ids = self.__prepare_image_ids(source_normalized_list, self.__batch_size, self.__device) if source_normalized_list else None
                
                # ==================== PREPARE MODEL INPUTS ====================
                if all_source_latents is not None:
                    model_input = torch.cat([packed_noisy_target, all_source_latents], dim=1)
                    model_img_ids = torch.cat([target_ids, source_ids], dim=1)
                else:
                    model_input = packed_noisy_target
                    model_img_ids = target_ids
                
                # Guidance scale (1.0 for training)
                guidance = torch.full(
                    (self.__batch_size,), 
                    config.GUIDANCE_SCALE, 
                    device=self.__device, 
                    dtype=torch.float32
                )
                
                # Timestep (already normalized)
                timestep = sigmas
                
                # ==================== FORWARD PASS ====================
                with torch.enable_grad():
                    model_output = self.__lora_dit(
                        hidden_states=model_input,           
                        encoder_hidden_states=prompt_embeds, 
                        txt_ids=text_ids,                    
                        img_ids=model_img_ids,               
                        timestep=timestep,
                        guidance=guidance,
                        return_dict=False,
                    )[0]
                
                # Extract ONLY target prediction (since output includes sources if concatenated)
                seq_target = packed_noisy_target.shape[1]
                pred_target = model_output[:, :seq_target, :]
                
                # ==================== COMPUTE LOSS ====================
                # Target velocity: v = noise - clean
                target_v = noise - clean_normalized
                packed_target_v = self.__pack_latents(target_v)
                
                # MSE loss on velocity prediction
                loss = F.mse_loss(
                    pred_target.float(), 
                    packed_target_v.float(), 
                    reduction='mean'
                )
                
                # ==================== BACKWARD PASS ====================
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.__lora_dit.parameters(), 
                    max_norm=1.0
                )
                
                self.__optimizer.step()
                scheduler.step()
                
                # ==================== LOGGING ====================
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "lr": f"{current_lr:.2e}"
                })
                
                epoch_loss += loss.item()
                
                # ==================== SAVE CHECKPOINTS ====================
                if (iteration + 1) % config.MODEL_SAVE_INTERATION == 0:
                    save_name = f"flux_lora_epoch_{epoch+1}_iter_{iteration+1}"
                    save_path = os.path.join(config.SAVE_PATH, save_name)
                    print(f"\n[CHECKPOINT] Saving LoRA to {save_path}...")
                    self.__lora_dit.save_pretrained(save_path)
                
                iteration += 1
            
            # ==================== EPOCH COMPLETE ====================
            average_loss = round(epoch_loss / len(dataloader), 4)
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{config.EPOCHS} Complete")
            print(f"Average Loss: {average_loss}")
            print(f"{'='*60}\n")
            
            # Save if minimum loss
            self.__save_min_loss_lora(loss=average_loss)
            
            # Log to file
            logger.log(loss=average_loss)
        
        # ==================== SAVE FINAL WEIGHTS ====================
        save_name = "lora_final_weights"
        save_path = os.path.join(config.SAVE_PATH, save_name)
        print(f"\n[FINAL] Saving final LoRA weights to {save_path}...")
        self.__lora_dit.save_pretrained(save_path)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print(f"Minimum loss achieved: {self.__min_loss}")
        print(f"Total iterations: {iteration}")
        print("="*60)
            
            
            
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FLUX 2 IMAGE-TO-IMAGE LORA TRAINING")
    print("(Using Precomputed Multimodal Embeddings)")
    print("="*60 + "\n")

    # Initialize trainer
    trainer = Flux2Trainer()

    # Load dataset
    print("[DATA] Loading dataset...")
    dataset = ImageDataset(
        data_dir=config.DATA_FOLDER, 
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WEIGHT)
    )
    print(f"[DATA] Dataset size: {len(dataset)} samples\n")

    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # Start training
    trainer.train(dataloader)