import torch

# Custom PyTorch-based re-identification model (no TensorFlow dependency)
class SimpleReIDModel(torch.nn.Module):
    def __init__(self, feature_dim=512):
        super(SimpleReIDModel, self).__init__()
        # Simple CNN for feature extraction
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = torch.nn.Linear(256, feature_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)