import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

NUM_CLASSES = 4  # adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

class ChestCancerDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): Директория с данными
            mode (string): 'train', 'test' или 'valid'
            transform (callable, optional): Трансформации для изображений
        """
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        
        self.classes = ['adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma', 'normal']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(mode):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def get_dataloaders(root_dir, batch_size=BATCH_SIZE):
    train_dataset = ChestCancerDataset(root_dir, mode='train', transform=get_transforms('train'))
    valid_dataset = ChestCancerDataset(root_dir, mode='valid', transform=get_transforms('valid'))
    test_dataset = ChestCancerDataset(root_dir, mode='test', transform=get_transforms('test'))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, valid_loader, test_loader

class ChestCancerModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, model_name='resnet50', pretrained=True):
        super(ChestCancerModel, self).__init__()
        
        # Загрузка предобученной модели
        if model_name == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        elif model_name == 'densenet121':
            self.model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes)
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Обучение
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass и оптимизация
            loss.backward()
            optimizer.step()
            
            # Статистика
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(valid_loader.dataset)
        epoch_acc = running_corrects.double() / len(valid_loader.dataset)
        
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Сохранение лучшей модели
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Best val Acc: {best_acc:.4f}')
    return model, history

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Test Accuracy: {acc:.4f}')
    
    # Confusion matrix и отчет о классификации
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                  target_names=['adenocarcinoma', 'large cell carcinoma', 
                                               'squamous cell carcinoma', 'normal'])
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    return acc, cm, report

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def run_pytorch_training(data_dir, model_name='resnet50', pretrained=True):
    train_loader, valid_loader, test_loader = get_dataloaders(data_dir)
    
    model = ChestCancerModel(num_classes=NUM_CLASSES, model_name=model_name, pretrained=pretrained)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    model, history = train_model(model, train_loader, valid_loader, criterion, optimizer)
    
    acc, cm, report = evaluate_model(model, test_loader)
    
    plot_training_history(history)
    
    return model, history, (acc, cm, report)

if __name__ == "__main__":
    data_dir = "path/to/data"
    model, history, metrics = run_pytorch_training(data_dir, model_name='resnet50', pretrained=True)