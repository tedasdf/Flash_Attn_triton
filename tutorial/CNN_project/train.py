from layer import Conv2d_layer
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class TritonMNISTDetector(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Backbone using your Triton layers
        self.conv1 = Conv2d_layer(1, 16, kernel_size=3, padding=1) # 28x28 -> 28x28
        self.conv2 = Conv2d_layer(16, 32, kernel_size=3, stride=2, padding=1) # 28x28 -> 14x14
        self.conv3 = Conv2d_layer(32, 64, kernel_size=3, stride=2, padding=1) # 14x14 -> 7x7
        
        # Detection Head: Output 15 channels (1 obj + 4 box + 10 class)
        # We use a 1x1 conv to map the 64 features to our 15 predictions
        self.detector = Conv2d_layer(64, 15, kernel_size=1)
        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # x is (B, 64, 7, 7)
        x = self.pool(x)     # (B, 64, 1, 1)
        x = torch.flatten(x, 1) # (B, 64)
        logits = self.fc(x)  # (B, 10)
        return logits
    


def create_localized_mnist(batch_size=32):
    # Load standard MNIST
    mnist = datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.ToTensor())
    loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
    
    for images, labels in loader:
        B = images.shape[0]
        # Create a 56x56 canvas (2x larger than MNIST)
        canvas = torch.zeros((B, 1, 56, 56), device='cuda')
        
        # Random top-left corners (0 to 28)
        top = torch.randint(0, 28, (B,))
        left = torch.randint(0, 28, (B,))
        
        target_boxes = torch.zeros((B, 4), device='cuda') # [x_center, y_center, w, h]
        
        for i in range(B):
            canvas[i, 0, top[i]:top[i]+28, left[i]:left[i]+28] = images[i]
            # Normalize coordinates to [0, 1] for the loss function
            target_boxes[i, 0] = (left[i] + 14) / 56.0  # x_center
            target_boxes[i, 1] = (top[i] + 14) / 56.0   # y_center
            target_boxes[i, 2] = 28 / 56.0              # width
            target_boxes[i, 3] = 28 / 56.0              # height
            
        yield canvas, target_boxes, labels.cuda()


import torch.nn.functional as F
      
# Initialize your Triton-powered detector
model = TritonMNISTDetector().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# In your training loop, it becomes very clean:
criterion = nn.CrossEntropyLoss()


print("Starting Training with Triton Kernels...")

for epoch in range(50):
    total_loss = 0
    # Use our generator for localized digits
    data_gen = create_localized_mnist(batch_size=32)
    
    for i in range(100):  # 100 steps per epoch
        images, boxes, labels = next(data_gen)

        optimizer.zero_grad()
        
        # 1. Forward Pass (im2col -> Grouped GEMM)
        # Output shape: (B, 15, 7, 7)
        # images: (B, 1, 56, 56), labels: (B)
        outputs = model(images)
  
        loss = F.cross_entropy(outputs, labels)

        # 3. Backward Pass (GEMM Transpose -> col2im)
        loss.backward()
        
        # 4. Update Weights
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch} | Loss: {total_loss/100:.4f}")

