# Bayesian R-LayerNorm - Colab Demo
# https://colab.research.google.com/github/yourusername/bayesian-r-layernorm

# @title Install Bayesian R-LayerNorm
!pip install bayesian-rlayernorm -q
!pip install gradio -q

# @title Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_rlayernorm import BayesianRLayerNorm, RLayerNorm, StandardLayerNorm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gradio as gr

# @title Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# @title Define Model with Bayesian R-LayerNorm
class SimpleCNN(nn.Module):
    def __init__(self, norm_type='bayesian_r_layer'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        if norm_type == 'bayesian':
            self.norm1 = BayesianRLayerNorm(32, lambda_init=0.01)
        elif norm_type == 'r_layer':
            self.norm1 = RLayerNorm(32, lambda_init=0.01)
        else:
            self.norm1 = StandardLayerNorm(32)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        if norm_type == 'bayesian':
            self.norm2 = BayesianRLayerNorm(64, lambda_init=0.01)
        elif norm_type == 'r_layer':
            self.norm2 = RLayerNorm(64, lambda_init=0.01)
        else:
            self.norm2 = StandardLayerNorm(64)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc = nn.Linear(64 * 8 * 8, 10)
    
    def forward(self, x, return_uncertainty=False):
        x = self.conv1(x)
        if hasattr(self.norm1, 'forward') and return_uncertainty:
            x, unc1 = self.norm1(x, return_uncertainty=True)
        else:
            x = self.norm1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        if hasattr(self.norm2, 'forward') and return_uncertainty:
            x, unc2 = self.norm2(x, return_uncertainty=True)
        else:
            x = self.norm2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if return_uncertainty:
            return x, (unc1, unc2)
        return x

# @title Test Different Normalization Methods
def compare_normalizations():
    # Create models
    models = {
        'Standard LayerNorm': SimpleCNN('layer'),
        'R-LayerNorm': SimpleCNN('r_layer'),
        'Bayesian R-LayerNorm': SimpleCNN('bayesian')
    }
    
    # Test with random noisy input
    x = torch.randn(4, 3, 32, 32).to(device)
    x_noisy = x + torch.randn_like(x) * 0.3  # Add noise
    
    results = {}
    for name, model in models.items():
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            if 'Bayesian' in name:
                output, uncertainty = model(x_noisy, return_uncertainty=True)
                uncertainty = uncertainty[0].mean().item()
            else:
                output = model(x_noisy)
                uncertainty = 0.0
            
            # Calculate output variance (measure of stability)
            output_var = output.var().item()
            
            results[name] = {
                'output_variance': output_var,
                'uncertainty': uncertainty
            }
    
    return results

# @title Gradio Demo
def process_image(image, noise_level, norm_type):
    # Convert PIL to tensor
    img = image.resize((32, 32))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    
    # Add noise
    noise = np.random.randn(*img_array.shape) * noise_level
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 1)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(noisy_img).unsqueeze(0).float()
    
    # Create model
    model = SimpleCNN(norm_type).eval()
    
    # Process
    with torch.no_grad():
        if norm_type == 'bayesian':
            output, uncertainty = model(img_tensor, return_uncertainty=True)
            uncertainty = uncertainty[0].mean().item()
        else:
            output = model(img_tensor)
            uncertainty = 0.0
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original vs noisy
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_img.transpose(1, 2, 0))
    axes[1].set_title(f'Noisy Image (level: {noise_level})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    return fig, f"Output shape: {output.shape}, Uncertainty: {uncertainty:.4f}"

# @title Create Interface
iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Slider(0, 0.5, value=0.1, label="Noise Level"),
        gr.Radio(['layer', 'r_layer', 'bayesian'], 
                value='bayesian', label="Normalization Type")
    ],
    outputs=[
        gr.Plot(label="Visualization"),
        gr.Textbox(label="Model Output")
    ],
    title="Bayesian R-LayerNorm Demo",
    description="Test different normalization methods on noisy images"
)

# @title Launch Demo
iface.launch(debug=True)
