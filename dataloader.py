from torch.utils.data import  Dataset
from glob import glob 
from torchvision import transforms
from PIL import Image 
import os 
import torch


class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size ):
        # Initialize dataset, e.g., list all image files in data_dir
        self.data_dir = data_dir
        
        self.transform =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize(mean=[0.5], std=[0.5]), # Map to [-1, 1]
    ])

        #load folders 
        self.__data = glob(f"{data_dir}/*")  # Example to load all subdirectories            
    
    def __len__(self):
        # Return the total number of samples
        return len(self.__data)
    
    def __image_to_tensor(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        # Load and return a sample from the dataset at the given index
        folder_path = self.__data[idx]

        input_images_paths = []
        original_image_path = os.path.join(folder_path, "input_original.png")
        # mask_image_path = os.path.join(folder_path, "input_mask.png")
        material_image_path = os.path.join(folder_path, "input_material.png")
        
        #create image list
        input_images_paths = [original_image_path, material_image_path]
        
        output_image_path = os.path.join(folder_path, "output.png")
        
        #load prompt and text embeds
        prompt_embeds_path = os.path.join(folder_path, "prompt_embeds.pt")
        text_ids_path = os.path.join(folder_path, "text_ids.pt")
        
        prompt_embeds = torch.load(prompt_embeds_path)
        text_ids = torch.load(text_ids_path)
                
        # 1. Load input images as PIL for the prompt encoder
        input_images_pil = []
        for img_path in input_images_paths:
            image = Image.open(img_path).convert("RGB")                
            input_images_pil.append(image)
        
        # 2. Load input images and transform to tensors for the VAE encoder
        input_images_tensors = [self.transform(img) for img in input_images_pil]
        
        # 3. Load and transform the output image to a tensor
        output_image = Image.open(output_image_path).convert("RGB")
        output_image_tensor = self.transform(output_image)
        
        sample = {
            'input_images_tensors': input_images_tensors, # For VAE
            'output_image_tensor': output_image_tensor,     # For VAE
            "prompt_embeds" : prompt_embeds,
            "text_ids" : text_ids
        }

        return sample