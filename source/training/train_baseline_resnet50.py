import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


from sklearn.metrics import confusion_matrix

def train_model():
    # --- CONFIG & DIRECTORIES ---
    MODEL_NAME = "resnet50_baseline"
    RESULTS_DIR = f"results/{MODEL_NAME}"
    CHARTS_DIR = f"results/charts/{MODEL_NAME}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- DATA LOADERS ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/processed/resized224p'
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_raw, val_raw = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_raw.dataset.transform = data_transforms['train']
    val_raw.dataset.transform = data_transforms['val']

    train_loader = DataLoader(train_raw, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_raw, batch_size=32, shuffle=False, num_workers=4)

    # --- MODEL, LOSS, OPTIMIZER ---
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # --- TRAINING LOOP ---
    num_epochs = 10
    history = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate Epoch Metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct / total
        
        print(f"Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # Record History
        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        })

        # Save BEST model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), f'models/{MODEL_NAME}_best.pth')
            print(f"New Best Model Saved with Accuracy: {best_acc:.2f}%")
    # --- LOGGING & PLOTTING ---
    df = pd.DataFrame(history)
    df.to_csv(f"{RESULTS_DIR}/metrics.csv", index=False)

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', color='blue')
    plt.title(f'{MODEL_NAME} Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.savefig(f"{CHARTS_DIR}/accuracy_chart.png")

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='red')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='orange')
    plt.title(f'{MODEL_NAME} Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{CHARTS_DIR}/loss_chart.png")

    # --- 5. CONFUSION MATRIX GENERATION ---
    print("\nGenerating Confusion Matrix...")
    model.load_state_dict(torch.load(f'models/{MODEL_NAME}_best.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating Best Model"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {MODEL_NAME}')
    plt.ylabel('True Artist')
    plt.xlabel('Predicted Artist')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/confusion_matrix.png")
    
    print(f"\nTraining complete. Results saved in {RESULTS_DIR}\nCharts saved in {CHARTS_DIR}\n\nModel saved in models")

if __name__ == "__main__":
    train_model()