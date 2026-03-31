from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import cv2
import os
from typing import List, Optional

class DINOv2Encoder:
    """
    DINOv2 Feature Encoder with batch processing support.
    Optimized for few-shot object recognition.
    """
    
    def __init__(self, model_size: str = "vits14", model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_size}_reg', pretrained=False)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_size}_reg')

        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        # Output dimension based on model size
        self._dim_map = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}
        self.embedding_dim = self._dim_map.get(model_size, 384)
    
    @torch.no_grad()
    def encode(self, cropped_image_bgr: np.ndarray) -> np.ndarray:
        """Encode single image to embedding vector."""
        rgb_image = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        embedding = self.model(input_tensor)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def encode_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Batch encode multiple images for efficiency.
        
        Args:
            images: List of BGR images (OpenCV format)
        
        Returns:
            np.ndarray of shape (N, embedding_dim)
        """
        if len(images) == 0:
            return np.empty((0, self.embedding_dim))
        
        tensors = []
        for img_bgr in images:
            rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform(pil_image)
            tensors.append(tensor)
        
        # Stack into batch
        batch = torch.stack(tensors).to(self.device)
        
        # Forward pass
        embeddings = self.model(batch)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    @torch.no_grad()
    def encode_with_augmentation(self, image_bgr: np.ndarray, num_augments: int = 3) -> np.ndarray:
        """
        Encode with test-time augmentation for more robust embedding.
        Uses multiple crops and averages the embeddings.
        
        Args:
            image_bgr: BGR image
            num_augments: Number of augmented views
        
        Returns:
            Averaged embedding vector
        """
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Create augmented views
        augment_transform = transforms.Compose([
            transforms.Resize(288, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        tensors = [self.transform(pil_image)]  # Original center crop
        for _ in range(num_augments - 1):
            tensors.append(augment_transform(pil_image))
        
        batch = torch.stack(tensors).to(self.device)
        embeddings = self.model(batch)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Average all augmented embeddings
        avg_embedding = embeddings.mean(dim=0, keepdim=True)
        avg_embedding = torch.nn.functional.normalize(avg_embedding, p=2, dim=1)
        
        return avg_embedding.cpu().numpy().flatten()
    
    def compute_similarity(self, query: np.ndarray, prototypes: dict) -> dict:
        """
        Compute similarity scores against all prototypes.
        
        Args:
            query: Query embedding vector
            prototypes: Dict of {name: embedding}
        
        Returns:
            Dict of {name: similarity_score}
        """
        scores = {}
        for name, proto in prototypes.items():
            sim = np.dot(query, proto) / (np.linalg.norm(query) * np.linalg.norm(proto) + 1e-10)
            scores[name] = float(sim)
        return scores