import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import matplotlib.pyplot as plt

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 0 — другие люди, 1 — вы
        for class_name, label in [('others', 0), ('me', 1)]:
            class_dir = self.root_dir / class_name
            print(f"Сканирование директории: {class_dir}")
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.image_paths.append(str(img_path))
                    self.labels.append(label)
                    
        print(f"Успешно загружено изображений: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
# Аугментация для тренировочной выборки
train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])

# Валидационные трансформации (только ресайз)
val_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Инициализируем датасеты по аналогии с вашей структурой путей
train_dataset = FaceDataset(root_dir='data/train', transform=train_transform)
val_dataset = FaceDataset(root_dir='data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class TinyFaceCNN(nn.Module):
    def __init__(self):
        super(TinyFaceCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, 2) # 2 класса: 0 — others, 1 — me
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, val_loader, epochs=5, lr=0.001):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        for epoch in range(epochs):
            # Режим обучения
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            
            # Режим валидации
            self.model.eval()
            running_val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    running_val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
            epoch_val_loss = running_val_loss / val_total
            epoch_val_acc = val_correct / val_total
            
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(epoch_val_acc)
            
            print(f"Epoch [{epoch+1}/{epochs}] — "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            
        return history
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TinyFaceCNN()
trainer = Trainer(model, device=device)

# Запускаем обучение на 10 эпох
history = trainer.train(train_loader, val_loader, epochs=5, lr=0.001)

# Отрисовка графиков в точности по вашему шаблону
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy')

plt.show()

dummy_input = torch.randn(1, 3, 96, 96).to(device)
onnx_path = "face_recognition_model.onnx"

torch.onnx.export(
    model.to(device), 
    dummy_input, 
    onnx_path, 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output']
)
print(f"Модель успешно сохранена в формат ONNX: {onnx_path}")