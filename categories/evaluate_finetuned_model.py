import logging
"""
EVALUATION SCRIPT для моделей Food101
=====================================

ІНСТРУКЦІЯ: Щоб вказати шлях до моделі, перейдіть до рядка ~270 і змініть:
model_path = "ваш_шлях_до_моделі.pth"

Приклади:
- model_path = "food101_densenet_finetuned_final_dataset_256_20250609_082804.pth"
- model_path = "food101_densenet_finetuned_final_20250607_142912.pth" 
- model_path = "c:/full/path/to/your/model.pth"

Скрипт генерує:
- confusion_matrix.png (візуальна матриця)
- confusion_matrix.csv (табличний формат)
- evaluation_results.csv (детальні метрики)
- top_confused_classes.png
- per_class_accuracy.png
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from collections import OrderedDict
import os
import time
from datetime import datetime

def load_model(model_path, num_classes, device):
    """Load the fine-tuned model"""
    logging.info(f"Loading model from {model_path}")
    
    # Create model architecture (same as training)
    model = models.densenet121(weights=None)  # Don't load pretrained weights
    
    # Get the number of input features for the classifier
    in_features = model.classifier.in_features
    
    # Create the same classifier head as used in training
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, 512)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(512, num_classes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
      # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    logging.info("Model loaded successfully!")
    return model

def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and return predictions and true labels"""
    logging.info("Starting model evaluation...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.exp(outputs)  # Convert log probabilities to probabilities
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if batch_idx % 20 == 0:
                logging.info(f'Processed batch {batch_idx+1}/{total_batches} ({100*batch_idx/total_batches:.1f}%)')
    
    logging.info("Model evaluation completed!")
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    logging.info("Creating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # For readability, if we have too many classes, show only top confused classes
    if len(class_names) > 20:
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   xticklabels=False, yticklabels=False)
        plt.title('Confusion Matrix (All Classes)', fontsize=16)
    else:
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Confusion matrix saved to {save_path}")
    plt.show()
    
    return cm

def plot_top_confused_classes(y_true, y_pred, class_names, top_n=15, save_path=None):
    """Plot top confused class pairs"""
    logging.info("Creating top confused classes plot...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Find top confused pairs (excluding diagonal)
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((class_names[i], class_names[j], cm[i, j]))
    
    # Sort by confusion count
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    top_confused = confused_pairs[:top_n]
    
    if top_confused:
        pairs = [f"{pair[0]} → {pair[1]}" for pair in top_confused]
        counts = [pair[2] for pair in top_confused]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(pairs, counts, color='coral')
        plt.xlabel('Number of Misclassifications')
        plt.title(f'Top {top_n} Most Confused Class Pairs')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(count), ha='left', va='center')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Top confused classes plot saved to {save_path}")
        plt.show()

def plot_class_accuracy(y_true, y_pred, class_names, save_path=None):
    """Plot per-class accuracy"""
    logging.info("Creating per-class accuracy plot...")
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Sort by accuracy
    sorted_indices = np.argsort(class_accuracy)
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_accuracy = class_accuracy[sorted_indices]
    
    plt.figure(figsize=(15, max(10, len(class_names) * 0.4)))
    colors = ['red' if acc < 0.5 else 'orange' if acc < 0.7 else 'green' for acc in sorted_accuracy]
    bars = plt.barh(range(len(sorted_classes)), sorted_accuracy, color=colors)
    
    plt.yticks(range(len(sorted_classes)), sorted_classes, fontsize=8)
    plt.xlabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xlim(0, 1)
    plt.grid(axis='x', alpha=0.3)
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, sorted_accuracy)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{acc:.3f}', ha='left', va='center', fontsize=7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Per-class accuracy plot saved to {save_path}")
    plt.show()
    
    return class_accuracy

def save_results_to_csv(y_true, y_pred, class_names, class_accuracy, save_path):
    """Save detailed results to CSV"""
    logging.info("Saving results to CSV...")
    
    # Calculate per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    # Overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    overall_f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Create detailed DataFrame
    results_data = []
    
    # Add overall metrics
    results_data.append({
        'Class': 'OVERALL',
        'Accuracy': overall_accuracy,
        'Precision': report['macro avg']['precision'],
        'Recall': report['macro avg']['recall'],
        'F1_Score': overall_f1_macro,
        'F1_Score_Weighted': overall_f1_weighted,
        'Support': int(report['macro avg']['support'])
    })
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        if class_name in report:
            results_data.append({
                'Class': class_name,
                'Accuracy': class_accuracy[i],
                'Precision': report[class_name]['precision'],
                'Recall': report[class_name]['recall'],
                'F1_Score': report[class_name]['f1-score'],
                'F1_Score_Weighted': overall_f1_weighted,  # Same for all classes
                'Support': int(report[class_name]['support'])
            })
    
    df = pd.DataFrame(results_data)
    df.to_csv(save_path, index=False)
    
    logging.info(f"Results saved to {save_path}")
    logging.info("\n=== OVERALL RESULTS ===")
    logging.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logging.info(f"Macro F1 Score: {overall_f1_macro:.4f}")
    logging.info(f"Weighted F1 Score: {overall_f1_weighted:.4f}")
    
    return df

def save_confusion_matrix_to_csv(y_true, y_pred, class_names, save_path):
    """Save confusion matrix to CSV file"""
    logging.info("Saving confusion matrix to CSV...")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create DataFrame with class names as both index and columns
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Add row and column totals
    cm_df['Total_Predicted'] = cm_df.sum(axis=1)
    cm_df.loc['Total_Actual'] = cm_df.sum(axis=0)
    
    # Save to CSV
    cm_df.to_csv(save_path)
    
    logging.info(f"Confusion matrix saved to {save_path}")
    return cm_df

def main():
    logging.info("=== Food101 Fine-tuned Model Evaluation ===")
    logging.info(f"Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
      # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")    
    # Paths - use original Food101 dataset (all categories)
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # ===================================================================
    # ВКАЗАТИ ШЛЯХ ДО МОДЕЛІ ТУТ (змініть на ваш файл .pth):
    # ===================================================================
    model_path = "../models/food_256_finetuned.pth"

    # Можливі варіанти:
    # model_path = "../models/food_256_finetuned.pth"       # Для dataset 256
    # model_path = "./food_256_finetuned.pth"               # Якщо модель в тій же папці
    # model_path = "models/food_256_finetuned.pth"          # Якщо модель в папці models
    # ===================================================================
    
    # Автоматичне визначення типу датасету з назви моделі
    model_filename = os.path.basename(model_path)
    if "dataset_256" in model_filename.lower():
        dataset_type = "256"
        expected_classes = 256
        logging.info("🎯 Виявлено модель для датасету 256 класів")
    else:
        dataset_type = "101"
        expected_classes = 101
        logging.info("🎯 Виявлено модель для датасету Food-101 (101 клас)")
      # Налаштування шляхів до датасету залежно від типу
    if dataset_type == "256":
        # Для датасету 256 класів потрібен специфічний шлях
        data_directory = 'dataset_256/images'  # Шлях до 256 датасету
        logging.info(f"🔍 Використовується датасет 256 класів з папки: {data_directory}")
    else:
        data_directory = 'dataset/images'  # Original Food101 dataset
        logging.info(f"🔍 Використовується оригінальний Food-101 датасет з папки: {data_directory}")
    
    # Створення абсолютного шляху до датасету
    absolute_data_dir = os.path.join(base_path, data_directory)
      # Check for dataset structure (category folders)
    if not os.path.exists(absolute_data_dir):
        # Fallback options for different dataset types
        if dataset_type == "256":
            fallback_paths = [
                'c:\\food_mobilenet\\food_mobilenet\\dataset_256\\images',
                'c:\\food_mobilenet\\food_mobilenet\\dataset_256',
                'c:\\food_mobilenet\\food_mobilenet\\dataset\\images'  # На випадок, якщо 256 в основній папці
            ]
        else:
            fallback_paths = [
                'c:\\food_mobilenet\\food_mobilenet\\dataset\\images',
                'c:\\food_mobilenet\\food_mobilenet\\dataset'
            ]
        
        found_dataset = False
        for fallback_path in fallback_paths:
            if os.path.exists(fallback_path):
                absolute_data_dir = fallback_path
                logging.info(f"✅ Знайдено fallback датасет: {absolute_data_dir}")
                found_dataset = True
                break
        
        if not found_dataset:
            logging.info(f"❌ Error: Не можу знайти датасет для типу '{dataset_type}'!")
            logging.info(f"Шукав за шляхами:")
            logging.info(f"  - {os.path.join(base_path, data_directory)}")
            for path in fallback_paths:
                logging.info(f"  - {path}")
            return
    
    logging.info(f"Using dataset directory: {absolute_data_dir}")
    
    # Перевірка, чи існує файл моделі
    if not os.path.isabs(model_path):
        # Якщо відносний шлях, то робимо абсолютний
        model_path = os.path.join(base_path, model_path)
    
    if not os.path.exists(model_path):
        logging.info(f"Error: Модель не знайдена за шляхом: {model_path}")
        logging.info("Будь ласка, вкажіть правильний шлях до моделі в коді.")
        return
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    logging.info(f"Використовується модель: {model_path}")
    
    # Data transforms (same as validation in training)
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    ])
      # Load entire dataset (all categories and images)
    try:
        logging.info(f"📂 Завантажуємо датасет з {dataset_type} класів...")
        logging.info(f"📁 Папка: {absolute_data_dir}")
        # Load all images from all categories directly
        full_dataset = datasets.ImageFolder(
            absolute_data_dir,  # Point directly to images folder with categories
            data_transforms
        )
        full_dataloader = DataLoader(
            full_dataset, 
            batch_size=64,  # Smaller batch size for evaluation
            shuffle=False, 
            num_workers=2,
            pin_memory=True        )        
        class_names = full_dataset.classes
        num_classes = len(class_names)
        logging.info(f"Number of classes: {num_classes}")
        logging.info(f"Number of total images: {len(full_dataset)}")
        logging.info(f"Classes found: {class_names[:10]}...")  # Show first 10 classes
        
        # Перевірка відповідності кількості класів
        if num_classes != expected_classes:
            logging.info(f"⚠️  УВАГА: Модель очікує {expected_classes} класів, а в датасеті знайдено {num_classes}")
            if dataset_type == "256" and num_classes < 256:
                logging.info("Можливо, потрібен інший датасет або шлях до папки з 256 категоріями")
            elif dataset_type == "101" and num_classes != 101:
                logging.info("Можливо, потрібен оригінальний Food-101 датасет")
            logging.info("Використовуємо кількість класів з моделі для загрузки архітектури...")
            model_num_classes = expected_classes
        else:
            logging.info(f"✅ Кількість класів співпадає: {num_classes}")
            model_num_classes = num_classes
        
    except Exception as e:
        logging.info(f"Error loading dataset: {e}")
        return
    
    # Load model
    try:
        model = load_model(model_path, model_num_classes, device)
    except Exception as e:
        logging.info(f"Error loading model: {e}")
        return
    
    # Evaluate model
    try:
        y_pred, y_true, y_probs = evaluate_model(model, full_dataloader, device, class_names)
    except Exception as e:
        logging.info(f"Error during evaluation: {e}")
        return
    
    # Create output directory for results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(base_path, f"evaluation_results_{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
      # Generate plots and save results
    try:
        # Confusion matrix (PNG)
        cm_path = os.path.join(results_dir, "confusion_matrix.png")
        cm = plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
        
        # Confusion matrix (CSV)
        cm_csv_path = os.path.join(results_dir, "confusion_matrix.csv")
        cm_df = save_confusion_matrix_to_csv(y_true, y_pred, class_names, cm_csv_path)
        
        # Top confused classes
        confused_path = os.path.join(results_dir, "top_confused_classes.png")
        plot_top_confused_classes(y_true, y_pred, class_names, save_path=confused_path)
        
        # Per-class accuracy
        accuracy_path = os.path.join(results_dir, "per_class_accuracy.png")
        class_accuracy = plot_class_accuracy(y_true, y_pred, class_names, accuracy_path)
          # Save results to CSV
        csv_path = os.path.join(results_dir, "evaluation_results.csv")
        results_df = save_results_to_csv(y_true, y_pred, class_names, class_accuracy, csv_path)
        
        logging.info(f"\nAll results saved to: {results_dir}")
        logging.info("Generated files:")
        logging.info("  - confusion_matrix.png (візуальна матриця)")
        logging.info("  - confusion_matrix.csv (табличний формат)")
        logging.info("  - evaluation_results.csv (детальні метрики)")
        logging.info("  - top_confused_classes.png")
        logging.info("  - per_class_accuracy.png")
        logging.info("Evaluation completed successfully!")
        
    except Exception as e:
        logging.info(f"Error generating results: {e}")

if __name__ == "__main__":
    main()
