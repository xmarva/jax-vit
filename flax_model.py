import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Any, Callable, Sequence, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

NUM_CLASSES = 4  # adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20
IMG_SIZE = 224

def preprocess_image(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    image = Image.open(image_path).convert('RGB').resize(target_size)
    image_array = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    return image_array.astype(np.float32)

def load_dataset(root_dir, mode='train'):
    data_dir = os.path.join(root_dir, mode)
    classes = ['adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma', 'normal']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    images = []
    labels = []
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        class_idx = class_to_idx[class_name]
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist")
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('.jpg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = preprocess_image(img_path)
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def get_batch_iterator(images, labels, batch_size=BATCH_SIZE, shuffle=True):
    num_samples = len(images)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_images = images[batch_indices]
        batch_labels = labels[batch_indices]
        
        yield batch_images, batch_labels

class ResNetBlock(nn.Module):
    features: int
    stride: int = 1
    
    @nn.compact
    def __call__(self, x, training=True):
        residual = x
        y = nn.Conv(self.features, kernel_size=(3, 3), strides=(self.stride, self.stride), 
                    padding=((1, 1), (1, 1)))(x)
        y = nn.BatchNorm(use_running_average=not training)(y)
        y = nn.relu(y)
        y = nn.Conv(self.features, kernel_size=(3, 3), padding=((1, 1), (1, 1)))(y)
        y = nn.BatchNorm(use_running_average=not training)(y)
        
        if residual.shape != y.shape:
            residual = nn.Conv(self.features, kernel_size=(1, 1), strides=(self.stride, self.stride))(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        return nn.relu(residual + y)

class ResNet(nn.Module):
    num_classes: int = NUM_CLASSES
    
    @nn.compact
    def __call__(self, x, training=True):
        # Initial layer
        x = nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding=((3, 3), (3, 3)))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        
        x = ResNetBlock(64)(x, training)
        x = ResNetBlock(64)(x, training)
        
        x = ResNetBlock(128, stride=2)(x, training)
        x = ResNetBlock(128)(x, training)
        
        x = ResNetBlock(256, stride=2)(x, training)
        x = ResNetBlock(256)(x, training)
        
        x = ResNetBlock(512, stride=2)(x, training)
        x = ResNetBlock(512)(x, training)
        
        # Global average pooling и классификация
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        
        return x

class MBConvBlock(nn.Module):
    expand_ratio: int
    output_channels: int
    stride: int = 1
    
    @nn.compact
    def __call__(self, x, training=True):
        input_channels = x.shape[-1]
        expanded_channels = input_channels * self.expand_ratio
        
        if self.expand_ratio != 1:
            expand = nn.Conv(expanded_channels, kernel_size=(1, 1))(x)
            expand = nn.BatchNorm(use_running_average=not training)(expand)
            expand = nn.swish(expand)
        else:
            expand = x
            
        depthwise = nn.Conv(
            expanded_channels, 
            kernel_size=(3, 3), 
            strides=(self.stride, self.stride), 
            padding=((1, 1), (1, 1)),
            feature_group_count=expanded_channels
        )(expand)
        depthwise = nn.BatchNorm(use_running_average=not training)(depthwise)
        depthwise = nn.swish(depthwise)
        
        output = nn.Conv(self.output_channels, kernel_size=(1, 1))(depthwise)
        output = nn.BatchNorm(use_running_average=not training)(output)
        
        if input_channels == self.output_channels and self.stride == 1:
            return x + output
        return output

class EfficientNetB0(nn.Module):
    num_classes: int = NUM_CLASSES
    
    @nn.compact
    def __call__(self, x, training=True):
        # Stem
        x = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.swish(x)
        
        x = MBConvBlock(1, 16)(x, training)
        
        x = MBConvBlock(6, 24, stride=2)(x, training)
        x = MBConvBlock(6, 24)(x, training)
        
        x = MBConvBlock(6, 40, stride=2)(x, training)
        x = MBConvBlock(6, 40)(x, training)
        
        x = MBConvBlock(6, 80, stride=2)(x, training)
        x = MBConvBlock(6, 80)(x, training)
        
        x = MBConvBlock(6, 112)(x, training)
        x = MBConvBlock(6, 112)(x, training)
        
        x = MBConvBlock(6, 192, stride=2)(x, training)
        x = MBConvBlock(6, 192)(x, training)
        
        x = MBConvBlock(6, 320)(x, training)
        
        x = nn.Conv(1280, kernel_size=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.swish(x)
        
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        
        return x
    
def create_model(model_name='resnet'):
    if model_name == 'resnet':
        return ResNet(num_classes=NUM_CLASSES)
    elif model_name == 'efficientnet':
        return EfficientNetB0(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def compute_accuracy(logits, labels):
    preds = jnp.argmax(logits, axis=1)
    return jnp.mean(preds == labels)

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, NUM_CLASSES)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=1))

def create_train_state(rng, model, learning_rate=LEARNING_RATE):
    params = model.init(rng, jnp.ones([1, IMG_SIZE, IMG_SIZE, 3]), training=True)
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch, rng):
    images, labels = batch
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images, training=True, rngs={'dropout': rng})
        loss = cross_entropy_loss(logits, labels)
        return loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = compute_accuracy(logits, labels)
    
    return state, loss, accuracy

@jax.jit
def eval_step(state, batch):
    images, labels = batch
    logits = state.apply_fn({'params': state.params}, images, training=False)
    loss = cross_entropy_loss(logits, labels)
    accuracy = compute_accuracy(logits, labels)
    
    return loss, accuracy, logits

def train_model_flax(state, train_images, train_labels, valid_images, valid_labels, num_epochs=NUM_EPOCHS):
    rng = jax.random.PRNGKey(0)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_params = None
    
    for epoch in range(num_epochs):
        rng, step_rng = jax.random.split(rng)
        train_losses, train_accs = [], []
        
        for batch in get_batch_iterator(train_images, train_labels, shuffle=True):
            rng, step_rng = jax.random.split(rng)
            state, loss, accuracy = train_step(state, batch, step_rng)
            train_losses.append(loss)
            train_accs.append(accuracy)
        
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        
        val_losses, val_accs = [], []
        all_logits, all_labels = [], []
        
        for batch in get_batch_iterator(valid_images, valid_labels, shuffle=False):
            loss, accuracy, logits = eval_step(state, batch)
            val_losses.append(loss)
            val_accs.append(accuracy)
            all_logits.append(logits)
            all_labels.append(batch[1])
        
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = state.params
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    state = train_state.TrainState.create(
        apply_fn=state.apply_fn,
        params=best_params,
        tx=state.tx
    )
    
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    return state, history

def evaluate_model_flax(state, test_images, test_labels):
    all_preds = []
    all_labels = []
    
    for batch in get_batch_iterator(test_images, test_labels, shuffle=False):
        _, _, logits = eval_step(state, batch)
        preds = jnp.argmax(logits, axis=1)
        
        all_preds.extend(preds)
        all_labels.extend(batch[1])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Метрики
    acc = np.mean(all_preds == all_labels)
    print(f"Test Accuracy: {acc:.4f}")