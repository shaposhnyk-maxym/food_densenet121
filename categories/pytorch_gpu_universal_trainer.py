import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from collections import OrderedDict
from torch.amp import autocast, GradScaler
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score
from datetime import datetime
import json
import argparse

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model when validation loss decreases"""
        self.best_weights = model.state_dict().copy()

def plot_training_history(history, phase_name, save_dir):
    """Plot training history for accuracy and loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history['train_acc'], label='Training Accuracy', marker='o')
    ax1.plot(history['val_acc'], label='Validation Accuracy', marker='s')
    ax1.set_title(f'{phase_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history['train_loss'], label='Training Loss', marker='o')
    ax2.plot(history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title(f'{phase_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{phase_name.lower().replace(" ", "_")}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    logging.info(f"Training history plot saved to: {save_path}")

def evaluate_model_performance(model, dataloader, device, class_names, phase_name, save_dir):
    """Evaluate model and create confusion matrix and metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    logging.info(f"Evaluating {phase_name} model performance...")
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    # Create evaluation results table
    results_data = {
        'Phase': [phase_name],
        'Accuracy': [accuracy],
        'F1_Score': [f1],
        'Recall': [recall],
        'Average': [(accuracy + f1 + recall) / 3]
    }
    
    results_df = pd.DataFrame(results_data)
    csv_path = os.path.join(save_dir, f'{phase_name.lower().replace(" ", "_")}_evaluation_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    if len(class_names) > 20:
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   xticklabels=False, yticklabels=False)
        plt.title(f'{phase_name} - Confusion Matrix (All Classes)')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{phase_name} - Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    cm_path = os.path.join(save_dir, f'{phase_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logging.info(f"\n=== {phase_name.upper()} EVALUATION RESULTS ===")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"Average: {(accuracy + f1 + recall) / 3:.4f}")
    logging.info(f"Results saved to: {csv_path}")
    logging.info(f"Confusion matrix saved to: {cm_path}")
    
    return results_df

def train_food101_model(data_dir, model_save_path, num_epochs=10, batch_size=32, learning_rate=0.001, 
                       model=None, is_finetuning=False, pretrained_weights_path=None, 
                       use_early_stopping=True, patience=15, phase_name="Training", save_dir=None):
    """
    Trains a DenseNet121 model on the Food101 dataset with early stopping and history tracking.

    Args:
        data_dir (str): Path to the dataset (expects 'train' and 'valid' subfolders).
        model_save_path (str): Path to save the best model.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and validation.
        learning_rate (float): Learning rate for the optimizer.
        model (torch.nn.Module, optional): Pre-trained model to continue training.
        is_finetuning (bool): Whether this is fine-tuning (unfreeze all layers) or initial training.
        pretrained_weights_path (str, optional): Path to load pretrained weights from.
        use_early_stopping (bool): Whether to use early stopping.
        patience (int): Early stopping patience.
        phase_name (str): Name of the training phase for visualization.
        save_dir (str): Directory to save plots and evaluation results.
    """
    logging.info(f"\n=== {phase_name.upper()} ===")
    logging.info(f"Data directory: {data_dir}")
    logging.info(f"Model save path: {model_save_path}")
    logging.info(f"Number of epochs: {num_epochs}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Early stopping: {use_early_stopping} (patience: {patience})")

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create save directory if not provided
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"training_results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'valid']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, 
                               num_workers=4, pin_memory=True, persistent_workers=True)
                       for x in ['train', 'valid']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
        class_names = image_datasets['train'].classes
        num_classes = len(class_names)
        logging.info(f"Number of classes: {num_classes}")
        logging.info(f"Training samples: {dataset_sizes['train']}")
        logging.info(f"Validation samples: {dataset_sizes['valid']}")
        if num_classes == 0:
            logging.info("Error: No classes found. Check your 'train' and 'valid' dataset folders.")
            return None, None
    except FileNotFoundError:
        logging.info(f"Error: Dataset directory not found at {data_dir}. Make sure 'train' and 'valid' subdirectories exist.")
        return None, None
    except Exception as e:
        logging.info(f"An error occurred while loading datasets: {e}")
        return None, None

    # Load or create model
    if model is None:
        # Load pre-trained DenseNet121 model
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # Get the number of input features for the classifier
        in_features = model.classifier.in_features

        # Create a new classifier head
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, 512)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(512, num_classes)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier

    # Load pretrained weights if provided
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        logging.info(f"Loading pretrained weights from {pretrained_weights_path}")
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))

    # Set up training parameters based on mode
    if is_finetuning:
        logging.info("Fine-tuning mode: Unfreezing all layers")
        # Unfreeze all parameters for fine-tuning
        for param in model.parameters():
            param.requires_grad = True
        # Use all model parameters for optimization
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        logging.info("Initial training mode: Only training classifier")
        # Freeze all parameters in the feature extraction part
        for param in model.parameters():
            param.requires_grad = False
        # Only make classifier parameters trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        # Only train the classifier parameters
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model = model.to(device)

    # Loss function
    criterion = nn.NLLLoss()

    # Learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, verbose=True)

    # Initialize early stopping
    if use_early_stopping:
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

    # Initialize GradScaler for AMP
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Training loop
    since = time.time()
    best_model_wts = model.state_dict()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass with autocast for mixed precision
                with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # Backward pass and optimization (only in training phase)
                if phase == 'train':
                    if device.type == 'cuda':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                # Clear cache to prevent OOM
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history for both phases
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

        # After both train and validation phases are complete for this epoch
        if len(history['val_loss']) > 0:  # Only proceed if validation phase completed
            current_val_loss = history['val_loss'][-1]
            
            # Deep copy the model if it's the best so far
            if current_val_loss < best_loss:
                best_loss = current_val_loss
                best_model_wts = model.state_dict().copy()
                torch.save(model.state_dict(), model_save_path)
                logging.info(f'Best model saved to {model_save_path} with validation loss: {best_loss:.4f}')

            # Check early stopping
            if use_early_stopping:
                if early_stopping(current_val_loss, model):
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    logging.info(f"Best validation loss: {early_stopping.best_loss:.4f}")
                    break

        # Step the scheduler
        scheduler.step(history['val_loss'][-1])

        print()

        # Break from outer loop if early stopping triggered
        if use_early_stopping and early_stopping.counter >= early_stopping.patience:
            break

    time_elapsed = time.time() - since
    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Loss: {best_loss:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot training history
    plot_training_history(history, phase_name, save_dir)

    # Evaluate model performance
    logging.info(f"\nEvaluating {phase_name} model...")
    evaluation_results = evaluate_model_performance(
        model, dataloaders['valid'], device, class_names, phase_name, save_dir
    )

    # Save training history
    history_path = os.path.join(save_dir, f'{phase_name.lower().replace(" ", "_")}_history.json')
    history_serializable = {k: [float(x) for x in v] for k, v in history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    logging.info(f"Training history saved to: {history_path}")

    return model, evaluation_results

def find_dataset_directory(dataset_type='regular'):
    """
    Автоматично знаходить правильний шлях до даних
    
    Args:
        dataset_type (str): 'regular' для звичайного датасету або '256' для dataset_256
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Варіанти шляхів для пошуку
    if dataset_type == '256':
        search_paths = [
            os.path.join(base_path, '..', 'dataset_256', 'images_splitted'),
            os.path.join(base_path, '..', 'dataset_256', 'images'),
            os.path.join(base_path, 'dataset_256', 'images_splitted'),
            os.path.join(base_path, 'dataset_256', 'images'),
        ]
    else:
        search_paths = [
            os.path.join(base_path, '..', 'dataset', 'images_splitted'),
            os.path.join(base_path, '..', 'dataset', 'images'),
            os.path.join(base_path, 'dataset', 'images_splitted'),
            os.path.join(base_path, 'dataset', 'images'),
        ]
    
    for path in search_paths:
        if os.path.exists(os.path.join(path, 'train')) and os.path.exists(os.path.join(path, 'valid')):
            logging.info(f"Found dataset at: {path}")
            return path
    
    logging.info(f"No valid dataset found for type '{dataset_type}'. Searched paths:")
    for path in search_paths:
        logging.info(f"  - {path}")
    return None

def run_two_phase_training(dataset_type='regular', phase1_epochs=None, phase2_epochs=None, 
                          batch_size=256, phase1_lr=0.001, phase2_lr=0.0001, patience=15):
    """
    Універсальна функція для двофазного навчання
    
    Args:
        dataset_type (str): 'regular' або '256' 
        phase1_epochs (int): Кількість епох для першої фази (за замовчуванням залежить від типу)
        phase2_epochs (int): Кількість епох для другої фази (за замовчуванням залежить від типу)
        batch_size (int): Розмір батчу
        phase1_lr (float): Learning rate для першої фази
        phase2_lr (float): Learning rate для другої фази
        patience (int): Терпіння для early stopping
    """
    # Record total training start time
    total_training_start = time.time()
    
    # Встановлюємо дефолтні значення епох залежно від типу датасету
    if phase1_epochs is None:
        phase1_epochs = 60 if dataset_type == '256' else 250
    if phase2_epochs is None:
        phase2_epochs = 40 if dataset_type == '256' else 150
    
    logging.info(f"=== UNIVERSAL FOOD101 TRAINER ===")
    logging.info(f"Dataset type: {dataset_type}")
    logging.info(f"Phase 1 epochs: {phase1_epochs}")
    logging.info(f"Phase 2 epochs: {phase2_epochs}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rates: {phase1_lr} -> {phase2_lr}")
    
    # Знаходимо датасет
    absolute_data_dir = find_dataset_directory(dataset_type)
    if absolute_data_dir is None:
        logging.info("Error: No valid dataset found. Please check your dataset structure.")
        return
    
    # Створюємо директорії для результатів
    base_path = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_path, f"universal_training_results_{dataset_type}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Шляхи збереження моделей
    model_save_phase1 = os.path.join(base_path, f"food101_densenet_head_trained_{dataset_type}_{timestamp}.pth")
    model_save_final = os.path.join(base_path, f"food101_densenet_finetuned_final_{dataset_type}_{timestamp}.pth")

    # Збираємо результати обох фаз
    all_results = []

    # ФАЗА 1: Навчання класифікатора
    logging.info("\n" + "=" * 60)
    logging.info("STARTING PHASE 1: HEAD TRAINING")
    logging.info("=" * 60)
    
    phase1_model, phase1_results = train_food101_model(
        data_dir=absolute_data_dir,
        model_save_path=model_save_phase1,
        num_epochs=phase1_epochs,
        batch_size=batch_size,
        learning_rate=phase1_lr,
        is_finetuning=False,
        use_early_stopping=True,
        patience=patience,
        phase_name="Phase 1 - Head Training",
        save_dir=results_dir
    )
    
    if phase1_model is None:
        logging.info("Phase 1 training failed.")
        return
    
    all_results.append(phase1_results)
    logging.info("Phase 1 completed successfully!")

    # ФАЗА 2: Дообучення всієї мережі
    logging.info("\n" + "=" * 60)
    logging.info("STARTING PHASE 2: FINE-TUNING")
    logging.info("=" * 60)
    
    phase2_model, phase2_results = train_food101_model(
        data_dir=absolute_data_dir,
        model_save_path=model_save_final,
        num_epochs=phase2_epochs,
        batch_size=batch_size,
        learning_rate=phase2_lr,
        is_finetuning=True,
        pretrained_weights_path=model_save_phase1,
        use_early_stopping=True,
        patience=patience,
        phase_name="Phase 2 - Fine-tuning",
        save_dir=results_dir
    )
    
    if phase2_model is None:
        logging.info("Phase 2 training failed.")
        return
    
    all_results.append(phase2_results)
    logging.info("Phase 2 completed successfully!")

    # Створюємо зведення результатів
    logging.info("\n" + "=" * 60)
    logging.info("TRAINING COMPLETE - CREATING SUMMARY")
    logging.info("=" * 60)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    summary_path = os.path.join(results_dir, "combined_training_summary.csv")
    combined_results.to_csv(summary_path, index=False)
    
    # Створюємо візуалізацію порівняння
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['Accuracy', 'F1_Score', 'Recall']
    phases = combined_results['Phase'].tolist()
    
    for i, metric in enumerate(metrics):
        values = combined_results[metric].tolist()
        bars = axes[i].bar(phases, values, color=['skyblue', 'lightcoral'])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].set_ylim(0, 1)
        
        # Додаємо значення на стовпці
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    summary_plot_path = os.path.join(results_dir, "training_phases_comparison.png")
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Підрахунок загального часу навчання
    total_training_time = time.time() - total_training_start
    total_hours = int(total_training_time // 3600)
    total_minutes = int((total_training_time % 3600) // 60)
    total_seconds = int(total_training_time % 60)
    
    # Створюємо зведення по часу
    timing_summary = {
        'dataset_type': dataset_type,
        'total_training_time_seconds': total_training_time,
        'total_training_time_formatted': f"{total_hours}h {total_minutes}m {total_seconds}s",
        'training_phases_completed': 2,
        'phase1_epochs': phase1_epochs,
        'phase2_epochs': phase2_epochs,
        'batch_size': batch_size,
        'timestamp_completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Зберігаємо інформацію про час
    timing_path = os.path.join(results_dir, "training_timing_summary.json")
    with open(timing_path, 'w') as f:
        json.dump(timing_summary, f, indent=2)
    
    logging.info(f"\n=== FINAL SUMMARY ===")
    logging.info(f"Dataset type: {dataset_type}")
    logging.info(combined_results.to_string(index=False))
    logging.info(f"\n=== TRAINING TIME SUMMARY ===")
    logging.info(f"Total Training Time: {timing_summary['total_training_time_formatted']}")
    logging.info(f"Total Training Time (seconds): {total_training_time:.2f}")
    logging.info(f"Training completed at: {timing_summary['timestamp_completed']}")
    logging.info(f"\nResults saved to: {results_dir}")
    logging.info(f"Summary CSV: {summary_path}")
    logging.info(f"Comparison plot: {summary_plot_path}")
    logging.info(f"Timing summary: {timing_path}")
    logging.info(f"Phase 1 model: {model_save_phase1}")
    logging.info(f"Final model: {model_save_final}")

def main():
    """Головна функція з підтримкою аргументів командного рядка"""
    parser = argparse.ArgumentParser(description='Universal Food101 Trainer')
    parser.add_argument('--dataset', choices=['regular', '256'], default='regular',
                      help='Dataset type to use (default: regular)')
    parser.add_argument('--phase1-epochs', type=int, default=None,
                      help='Number of epochs for phase 1 (default: auto based on dataset)')
    parser.add_argument('--phase2-epochs', type=int, default=None,
                      help='Number of epochs for phase 2 (default: auto based on dataset)')
    parser.add_argument('--batch-size', type=int, default=256,
                      help='Batch size for training (default: 256)')
    parser.add_argument('--phase1-lr', type=float, default=0.001,
                      help='Learning rate for phase 1 (default: 0.001)')
    parser.add_argument('--phase2-lr', type=float, default=0.0001,
                      help='Learning rate for phase 2 (default: 0.0001)')
    parser.add_argument('--patience', type=int, default=15,
                      help='Early stopping patience (default: 15)')
    
    args = parser.parse_args()
    
    run_two_phase_training(
        dataset_type=args.dataset,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        patience=args.patience
    )

if __name__ == "__main__":
    # Якщо скрипт запущено без аргументів, показуємо меню
    import sys
    if len(sys.argv) == 1:
        logging.info("=== UNIVERSAL FOOD101 TRAINER ===")
        logging.info("Виберіть тип датасету:")
        logging.info("1. Звичайний датасет (dataset/)")
        logging.info("2. Датасет 256 (dataset_256/)")
        logging.info("3. Запустити з параметрами по замовчуванню для звичайного датасету")
        logging.info("4. Запустити з параметрами по замовчуванню для датасету 256")
        
        choice = input("Ваш вибір (1-4): ").strip()
        
        if choice == '1':
            run_two_phase_training(dataset_type='regular')
        elif choice == '2':
            run_two_phase_training(dataset_type='256')
        elif choice == '3':
            run_two_phase_training(dataset_type='regular')
        elif choice == '4':
            run_two_phase_training(dataset_type='256')
        else:
            logging.info("Неправильний вибір. Використовуйте --help для допомоги з аргументами.")
    else:
        main()
