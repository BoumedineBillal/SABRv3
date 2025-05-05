# SABR: Sparsity-Accuracy Balanced Regularization

SABR (Sparsity-Accuracy Balanced Regularization) is a neural network pruning approach that balances model sparsity and accuracy through adaptive regularization.

## Project Overview

This project implements the SABR algorithm for optimizing neural networks by systematically inducing sparsity while maintaining accuracy. The implementation follows a three-phase approach:

1. **Base model training**: Standard training without regularization
2. **SABR regularization**: Applying adaptive L1 regularization with progressive parameter freezing
3. **Post-training pruning**: Converting near-zero weights to exact zeros for deployment

## Project Structure

```
SABRv2/
│
├── sabr/                         # Core SABR algorithm 
│   ├── __init__.py               # Package initialization
│   ├── sabr.py                   # Main SABR algorithm
│   ├── lambda_calculator.py      # Lambda calculation functions
│   └── pruning.py                # Pruning functionality
│
├── utils/                        # Non-essential utilities
│   ├── __init__.py
│   ├── visualization.py          # Plotting functions
│   ├── data_handler.py           # Dataset utilities
│   └── training_utils.py         # Training helpers
│
├── main.py                       # Complete training workflow
├── realtime_recognition.py       # Real-time testing application
└── README.md                     # This file
```

## How to Use SABR

The `main.py` file shows a complete workflow using SABR. Here's how to use it in your own projects:

### 1. Base Model Training

```python
from torchvision import models
import torch.nn as nn

# Define your model
model = models.vgg11(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes),
    nn.Softmax(dim=1)
)

# Train the base model using standard techniques
train_base_model(model, train_loader, val_loader, num_epochs=15)
```

### 2. SABR Regularization

```python
from sabr.sabr import SABR
from sabr.pruning import eliminate_dropout

# Eliminate dropout after base training
eliminate_dropout(model)

# Create SABR regularizer
regularizer = SABR(
    model,
    teta1=5e-4,              # Multiplier for std-based lambda initialization
    growth_rate=1.15,        # Factor to increase lambda values (15%)
    decrease_rate=1.2,       # Factor to decrease lambda values (20%)
    accuracy_threshold=0.007, # Maximum allowed accuracy drop
    k_epoch=2                # Epochs before freezing most sparse parameter
)

# Training loop with SABR
for epoch in range(num_epochs):
    # Training phase
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Add L1 penalty from SABR
        l1_penalty = regularizer.get_penalty(model)
        loss = loss + l1_penalty
        
        loss.backward()
        optimizer.step()
    
    # Validation phase
    val_loss, val_acc = validate_model()
    
    # Update lambda values based on validation accuracy
    regularizer.update(val_acc)
    
    # Check if all parameters are frozen
    if regularizer.all_parameters_frozen():
        print("All parameters have been frozen. Stopping training.")
        break
```

### 3. Post-Training Pruning

```python
from sabr.pruning import prune_model

# Apply pruning to convert near-zero weights to exact zeros
pruned_model, pruning_stats = prune_model(
    model,
    use_std_based=True,  # Use standard deviation-based thresholds
    teta1=0.2,           # Multiplier for standard deviation
    gamma=0.1            # Filter weights below gamma*std when calculating std
)

# Save the pruned model
torch.save(pruned_model.state_dict(), 'pruned_model.pth')
```

## Running the Example

1. **Install Dependencies**

```bash
pip install torch torchvision numpy matplotlib pandas tqdm pillow opencv-python scikit-learn
```

2. **Run the Complete Pipeline**

```bash
python main.py
```

3. **Test with Real-time Recognition**

```bash
python realtime_recognition.py --model results/pruned_model/pruned_model_std_based_teta1_0.2_gamma_0.1.pth
```

## Customization

- Adjust `teta1` to control the initial regularization strength
- Modify `growth_rate` and `decrease_rate` to change how quickly lambdas are adjusted
- Change `accuracy_threshold` to balance sparsity vs. accuracy preservation
- Set `k_epoch` to control how quickly parameters are frozen

## Key Components

- **SABR Class**: Core algorithm for adaptive regularization
- **Lambda Calculator**: Computes layer-specific lambda values
- **Pruning Module**: Converts near-zero weights to exact zeros
- **Visualization Utilities**: Plot training progress and results

## Real-time Inference

The included real-time inference application demonstrates the pruned model in action using webcam input. It supports:

- Region-of-interest (ROI) selection for targeted recognition
- Reference image display for each recognized gesture
- FPS monitoring and frame capture
