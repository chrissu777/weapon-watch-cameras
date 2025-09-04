import torch
import cv2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Simple histogram-based feature extractor that provides more discriminative features
class SimpleReIDModel(torch.nn.Module):
    def __init__(self, feature_dim=512):
        super(SimpleReIDModel, self).__init__()
        self.feature_dim = feature_dim
        
    def forward(self, x):
        # Convert tensor to numpy for traditional CV feature extraction
        if x.is_cuda:
            x_np = x.cpu().detach().numpy()
        else:
            x_np = x.detach().numpy()
        
        features_list = []
        
        for i in range(x_np.shape[0]):  # Process each image in batch
            img = x_np[i]
            
            # Convert from normalized tensor format back to 0-255 image
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalize
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            # Extract multiple types of features
            features = []
            
            # 1. Color histograms (RGB)
            for channel in range(3):
                hist = cv2.calcHist([img], [channel], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # 2. HSV color features
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            for channel in range(3):
                hist = cv2.calcHist([hsv], [channel], None, [16], [0, 256])
                features.extend(hist.flatten())
            
            # 3. Texture features using LBP-like approach
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Simple texture patterns
            height, width = gray.shape
            texture_features = []
            
            # Horizontal differences
            h_diff = np.abs(gray[:, 1:] - gray[:, :-1])
            texture_features.extend([np.mean(h_diff), np.std(h_diff)])
            
            # Vertical differences  
            v_diff = np.abs(gray[1:, :] - gray[:-1, :])
            texture_features.extend([np.mean(v_diff), np.std(v_diff)])
            
            # Intensity statistics
            texture_features.extend([np.mean(gray), np.std(gray), np.median(gray)])
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            texture_features.append(np.sum(edges > 0) / (height * width))
            
            features.extend(texture_features)
            
            # 4. Spatial features (position-based)
            # Divide image into 4x2 grid and compute mean intensities
            h_step, w_step = height // 4, width // 2
            for hi in range(4):
                for wi in range(2):
                    roi = gray[hi*h_step:(hi+1)*h_step, wi*w_step:(wi+1)*w_step]
                    if roi.size > 0:
                        features.append(np.mean(roi))
                    else:
                        features.append(0)
            
            # Pad or truncate to desired feature dimension
            features = np.array(features)
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            elif len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)), 'constant')
            
            # Normalize features
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            features_list.append(features)
        
        # Convert back to tensor
        result = torch.tensor(np.array(features_list), dtype=torch.float32)
        if x.is_cuda:
            result = result.cuda()
        
        return result