import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim import lr_scheduler
from datetime import datetime
import warnings
import sys
import re

# --- Налаштування ---
# Вимикаємо попередження від PIL
warnings.filterwarnings("ignore", "(?i)Corrupt EXIF data")

# --- Глобальні параметри ---
# Шляхи адаптовані для структури репозиторію food_densenet_121
FILE_RECIPES = '../data/recipes_dataset_en_cleaned.json'
FILE_IMAGE_ASSIGNMENTS = '../data/image_to_recipe_assignments_f4_its.json'
DATASET_ROOT_PATH = '../../datasets/food256/'  # Food256 датасет на рівень вище
MIN_INGREDIENT_FREQUENCY = 20
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10
NUM_WORKERS = 4

# --- КАРТА КОНСОЛІДАЦІЇ (V4/V5) ---
CONSOLIDATION_MAP = {
    'all purpose flour': 'flour', 'all-purpose flour': 'flour', 'bread flour': 'flour', 'cake flour': 'flour',
    'self-rising flour': 'flour',
    'granulated sugar': 'sugar', 'brown sugar': 'sugar', 'white sugar': 'sugar', 'confectioners sugar': 'sugar',
    'powdered sugar': 'sugar', 'caster sugar': 'sugar',
    'kosher salt': 'salt', 'sea salt': 'salt', 'table salt': 'salt',
    'black pepper': 'pepper', 'white pepper': 'pepper', 'cayenne pepper': 'cayenne',
    'red pepper flakes': 'red pepper flakes', 'chili powder': 'chili powder',
    'olive oil': 'oil', 'vegetable oil': 'oil', 'canola oil': 'oil', 'sesame oil': 'sesame oil', 'peanut oil': 'oil',
    'coconut oil': 'coconut oil',
    'unsalted butter': 'butter', 'salted butter': 'butter',
    'yellow onion': 'onion', 'white onion': 'onion', 'red onion': 'onion', 'green onion': 'scallions',
    'spring onion': 'scallions', 'scallion': 'scallions', 'shallot': 'shallots',
    'parmesan cheese': 'parmesan', 'parmigiano reggiano': 'parmesan', 'parmesan': 'parmesan',
    'pecorino cheese': 'pecorino', 'pecorino romano': 'pecorino', 'romano cheese': 'pecorino',
    'cheddar cheese': 'cheddar', 'sharp cheddar': 'cheddar', 'cheddar': 'cheddar',
    'mozzarella cheese': 'mozzarella', 'mozzarella': 'mozzarella',
    'goat cheese': 'goat cheese', 'feta cheese': 'feta', 'feta': 'feta', 'blue cheese': 'blue cheese',
    'gruyere': 'gruyere', 'swiss cheese': 'swiss cheese', 'ricotta': 'ricotta', 'mascarpone': 'mascarpone',
    'cream cheese': 'cream cheese', 'sour cream': 'sour cream', 'heavy cream': 'cream', 'whipping cream': 'cream',
    'chicken breast': 'chicken', 'chicken thigh': 'chicken', 'chicken wing': 'chicken',
    'ground beef': 'beef', 'beef steak': 'beef', 'beef tenderloin': 'beef', 'filet mignon': 'beef',
    'pork chop': 'pork', 'ground pork': 'pork', 'pork tenderloin': 'pork',
    'egg yolk': 'egg', 'egg white': 'egg',
    'roma tomato': 'tomato', 'cherry tomato': 'tomato',
    'russet potato': 'potato', 'yukon gold potato': 'potato',
    'lemon juice': 'lemon', 'lemon zest': 'lemon',
    'lime juice': 'lime', 'lime zest': 'lime',
    'soy sauce': 'soy sauce', 'fish sauce': 'fish sauce', 'oyster sauce': 'oyster sauce',
    'hoisin sauce': 'hoisin sauce', 'teriyaki sauce': 'teriyaki sauce', 'bbq sauce': 'bbq sauce',
    'hot sauce': 'hot sauce', 'sriracha': 'sriracha', 'worcestershire sauce': 'worcestershire sauce',
    'dijon mustard': 'mustard', 'yellow mustard': 'mustard',
    'baking soda': 'baking soda', 'baking powder': 'baking powder',
    'vanilla extract': 'vanilla', 'vanilla bean': 'vanilla',
    'red wine vinegar': 'vinegar', 'white wine vinegar': 'vinegar', 'apple cider vinegar': 'vinegar',
    'balsamic vinegar': 'vinegar', 'rice vinegar': 'vinegar',
    'red wine': 'wine', 'white wine': 'wine',
    'bread crumb': 'breadcrumbs', 'panko': 'breadcrumbs',
    'rice noodle': 'noodles', 'egg noodle': 'noodles', 'ramen': 'noodles', 'spaghetti': 'pasta', 'macaroni': 'pasta',
    'penne': 'pasta', 'lasagna': 'pasta',
    'sushi rice': 'rice', 'basmati rice': 'rice', 'jasmine rice': 'rice', 'arborio rice': 'rice', 'brown rice': 'rice',
    'white rice': 'rice',
    'black bean': 'beans', 'kidney bean': 'beans', 'pinto bean': 'beans', 'garbanzo bean': 'chickpeas',
    'chickpea': 'chickpeas',
    'flour': 'flour', 'sugar': 'sugar', 'salt': 'salt', 'pepper': 'pepper', 'oil': 'oil', 'butter': 'butter',
    'onion': 'onion', 'garlic': 'garlic', 'cheese': 'cheese', 'chicken': 'chicken', 'beef': 'beef', 'pork': 'pork',
    'egg': 'egg', 'eggs': 'egg', 'tomato': 'tomato', 'tomatoes': 'tomato', 'potato': 'potato', 'potatoes': 'potato',
    'lemon': 'lemon', 'lime': 'lime', 'water': 'water', 'milk': 'milk', 'cream': 'cream', 'vinegar': 'vinegar',
    'sauce': 'sauce', 'powder': 'powder', 'zucchini': 'zucchini', 'crouton': 'croutons', 'croutons': 'croutons',
    'breadcrumb': 'breadcrumbs', 'breadcrumbs': 'breadcrumbs', 'shrimp': 'shrimp', 'cilantro': 'cilantro',
    'parsley': 'parsley',
    'basil': 'basil', 'thyme': 'thyme', 'rosemary': 'rosemary', 'oregano': 'oregano', 'mint': 'mint',
    'cinnamon': 'cinnamon', 'cumin': 'cumin', 'coriander': 'coriander', 'paprika': 'paprika', 'turmeric': 'turmeric',
    'ginger': 'ginger', 'nutmeg': 'nutmeg', 'clove': 'cloves', 'cloves': 'cloves', 'cayenne': 'cayenne',
    'carrot': 'carrot', 'carrots': 'carrot', 'celery': 'celery', 'bell pepper': 'bell pepper', 'chili': 'chili',
    'chile': 'chili',
    'rice': 'rice', 'bean': 'beans', 'beans': 'beans', 'pea': 'peas', 'peas': 'peas', 'noodle': 'noodles',
    'noodles': 'noodles', 'pasta': 'pasta',
    'wine': 'wine', 'mustard': 'mustard', 'ketchup': 'ketchup', 'mayonnaise': 'mayonnaise', 'mayo': 'mayonnaise',
    'honey': 'honey', 'maple syrup': 'maple syrup', 'vanilla': 'vanilla', 'yeast': 'yeast', 'cornstarch': 'cornstarch',
    'tahini': 'tahini', 'sesame': 'sesame', 'almond': 'almonds', 'almonds': 'almonds', 'walnut': 'walnuts',
    'walnuts': 'walnuts',
    'pecan': 'pecans', 'pecans': 'pecans', 'peanut': 'peanuts', 'peanuts': 'peanuts', 'cashew': 'cashews',
    'pistachio': 'pistachios',
    'scallions': 'scallions', 'shallots': 'shallots', 'leek': 'leeks', 'leeks': 'leeks',
    'mushroom': 'mushrooms', 'mushrooms': 'mushrooms', 'spinach': 'spinach', 'kale': 'kale', 'arugula': 'arugula',
    'lettuce': 'lettuce',
    'cabbage': 'cabbage', 'broccoli': 'broccoli', 'cauliflower': 'cauliflower', 'avocado': 'avocado',
    'cucumber': 'cucumber',
    'eggplant': 'eggplant', 'squash': 'squash', 'pumpkin': 'pumpkin', 'apple': 'apple', 'apples': 'apple',
    'banana': 'banana', 'orange': 'orange', 'strawberry': 'strawberries', 'strawberries': 'strawberries',
    'blueberry': 'blueberries', 'blueberries': 'blueberries', 'raspberry': 'raspberries', 'raspberries': 'raspberries',
    'blackberry': 'blackberries', 'blackberries': 'blackberries', 'berry': 'berries', 'berries': 'berries',
    'grape': 'grapes', 'grapes': 'grapes', 'mango': 'mango', 'pineapple': 'pineapple', 'coconut': 'coconut',
    'olive': 'olives', 'olives': 'olives', 'tofu': 'tofu', 'salmon': 'salmon', 'tuna': 'tuna', 'cod': 'cod',
    'fish': 'fish', 'crab': 'crab', 'lobster': 'lobster', 'scallop': 'scallops', 'scallops': 'scallops',
    'clam': 'clams', 'clams': 'clams', 'mussel': 'mussels', 'mussels': 'mussels', 'oyster': 'oysters',
    'oysters': 'oysters',
    'seafood': 'seafood', 'bacon': 'bacon', 'sausage': 'sausage', 'ham': 'ham', 'lamb': 'lamb', 'turkey': 'turkey',
    'duck': 'duck', 'vodka': 'vodka', 'rum': 'rum', 'brandy': 'brandy', 'whiskey': 'whiskey', 'tequila': 'tequila',
    'gin': 'gin', 'beer': 'beer', 'bread': 'bread', 'tortilla': 'tortillas', 'tortillas': 'tortillas',
    'chocolate': 'chocolate', 'cocoa': 'cocoa', 'coffee': 'coffee', 'tea': 'tea', 'yogurt': 'yogurt',
    'ice cream': 'ice cream', 'gelatin': 'gelatin', 'stock': 'stock', 'broth': 'broth', 'bouillon': 'bouillon'
}

# --- СТОП-СЛОВА ---
STOP_WORDS = {
    'cup', 'cups', 'c', 'tbsp', 'tablespoon', 'tablespoons', 'tsp', 'teaspoon', 'teaspoons',
    'oz', 'ounce', 'ounces', 'lb', 'lbs', 'pound', 'pounds', 'g', 'kg', 'gram', 'grams',
    'ml', 'liter', 'liters', 'cm', 'inch', 'piece', 'pieces', 'clove', 'cloves',
    'chopped', 'diced', 'minced', 'sliced', 'finely', 'roughly', 'large', 'small', 'medium',
    'peeled', 'seeded', 'fresh', 'dried', 'freshly', 'ground', 'packed', 'cut',
    'optional', 'to', 'taste', 'or', 'and', 'as', 'needed', 'divided', 'melted', 'softened',
    'into', 'thin', 'thick', 'strips', 'cubes', 'room', 'temperature', 'about', 'garnish',
    'beaten', 'sifted', 'plus', 'more', 'for', 'at', 'drained', 'rinsed', 'crushed',
    'such', 'uncooked', 'cooked', 'warm', 'cold', 'hot', 'of', 'with', 'in', 'on', 'hand',
    'deveined', 'shelled', 'halved', 'quartered', 'julienned', 'wedges', 'batons', 'matchsticks',
    'package', 'container', 'bottle', 'can', 'jar', 'box', 'bag', 'stalk', 'stalks', 'head', 'bunch', 'sprig', 'sprigs',
    'handful', 'pinch', 'dash', 'recipe', 'follows', 'homemade', 'style', 'e', 'g', 'prepared',
    'without', 'stems', 'leaves', 'leaf', 'serve', 'serving', 'extra', 'virgin', 'vegan',
    'substitute', 'skinless', 'boneless', 'natural', 'organic', 'low', 'sodium', 'fat', 'free',
    'reduced', 'light', 'dark', 'yellow', 'white', 'red', 'green', 'black', 'brown', 'golden'
}

# Сортуємо ключі для правильного матчингу
SORTED_CONSOLIDATION_KEYS = sorted(CONSOLIDATION_MAP.keys(), key=len, reverse=True)

# --- Клас EarlyStopping ---
class EarlyStopping:
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
            return False
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logging.info("EarlyStopping: Зупинка тренування.")
                return True
            return False

    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict()
            logging.info("EarlyStopping: Збережено нові найкращі ваги.")

    def restore_model(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logging.info("EarlyStopping: Відновлено найкращі ваги моделі.")


# --- Функція завантаження та об'єднання даних (V5 - Whitelist & Caching) ---
def load_and_prepare_data(recipe_file, image_file, min_freq):
    logging.info(f"Завантаження файлу рецептів: {recipe_file}")
    df_recipes = pd.read_json(recipe_file)
    logging.info(f"Завантажено {len(df_recipes)} рецептів.")

    # --- Парсер ---
    def parse_ingredient(ing_str):
        ing_str = str(ing_str).lower()
        ing_str = re.sub(r'\([^)]*\)', '', ing_str)
        ing_str = re.sub(r'[\d½¼¾⅓⅔⅛\.\-\/]+', ' ', ing_str)
        ing_str = re.sub(r'[^\w\s]', '', ing_str)

        words = ing_str.split()
        cleaned_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
        cleaned_ing = ' '.join(cleaned_words).strip()

        if not cleaned_ing:
            return None

        for base_name in SORTED_CONSOLIDATION_KEYS:
            if re.search(r'\b' + re.escape(base_name) + r'\b', cleaned_ing):
                return CONSOLIDATION_MAP[base_name]
        return None

    # --- Кінець парсера ---

    df_recipes['cleaned_ingredients'] = df_recipes['ingredients'].apply(
        lambda ing_list: list(set(parse_ingredient(ing) for ing in ing_list if parse_ingredient(ing)))
    )
    recipe_ingredient_map = df_recipes.set_index('recipe_name')['cleaned_ingredients'].to_dict()
    logging.info("Очищено та КОНСОЛІДОВАНО (V5) інгредієнти в рецептах.")

    logging.info(f"Завантаження файлу зіставлення зображень: {image_file}")
    df_images = pd.read_json(image_file)
    logging.info(f"Завантажено {len(df_images)} зіставлень зображень.")

    df_images['ingredients'] = df_images['best_match_recipe_name'].map(recipe_ingredient_map)

    original_count = len(df_images)
    df_images = df_images.dropna(subset=['ingredients'])
    df_images = df_images[df_images['ingredients'].map(len) > 0]
    logging.info(f"Видалено {original_count - len(df_images)} зображень без відповідного / дійсного рецепта.")

    def clean_path(p):
        if 'dataset_256/images' in p:
            parts = p.split('dataset_256/images')
            rel_path = 'dataset_256/images' + parts[-1]
            return rel_path.replace('\\', '/')
        return None

    df_images['relative_path'] = df_images['image_path'].apply(clean_path)
    df_images = df_images.dropna(subset=['relative_path'])
    logging.info(f"Очищено шляхи до зображень. Залишилось {len(df_images)} зображень.")

    # --- Створення або завантаження словника ---
    vocab_file = "../models/ingredient_vocabulary_V4_FINAL.json"

    if os.path.exists(vocab_file):
        logging.info(f"Завантаження існуючого словника з {vocab_file}...")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            ingredient_to_index = json.load(f)
        num_classes = len(ingredient_to_index)
        logging.info(f"Завантажено {num_classes} інгредієнтів.")
    else:
        logging.info(f"Створення нового словника (файл {vocab_file} не знайдено)...")
        all_ingredients = [ing for sublist in df_images['ingredients'] for ing in sublist]
        ingredient_counts = Counter(all_ingredients)
        logging.info(f"Всього знайдено {len(ingredient_counts)} унікальних очищених інгредієнтів.")

        vocabulary = sorted([
            ing for ing, count in ingredient_counts.items() if count >= min_freq
        ])
        num_classes = len(vocabulary)
        logging.info(f"Після фільтрації (мін. частота = {min_freq}): {num_classes} інгредієнтів у словнику.")

        if num_classes == 0:
            raise ValueError("Словник інгредієнтів порожній.")

        ingredient_to_index = {ing: i for i, ing in enumerate(vocabulary)}

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(ingredient_to_index, f, indent=4, ensure_ascii=False)
        logging.info(f"НОВИЙ ФІНАЛЬНИЙ (V5) словник інгредієнтів збережено у {vocab_file}")

    # --- Розрахунок pos_weight ---
    logging.info("Розрахунок ваги для позитивного класу (pos_weight)...")
    total_positives = 0
    for ingredients_list in df_images['ingredients']:
        for ing in ingredients_list:
            if ing in ingredient_to_index:
                total_positives += 1

    total_elements = len(df_images) * num_classes
    total_negatives = total_elements - total_positives

    if total_positives == 0:
        pos_weight = 1.0
    else:
        pos_weight = total_negatives / total_positives

    logging.info(f"Розрахована pos_weight: {pos_weight:.2f}")

    return df_images[['relative_path', 'ingredients']], ingredient_to_index, num_classes, pos_weight


# --- Клас PyTorch Dataset ---
class IngredientDataset(Dataset):
    def __init__(self, dataframe, ingredient_to_index, num_classes, dataset_root, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.ingredient_to_index = ingredient_to_index
        self.num_classes = num_classes
        self.dataset_root = dataset_root
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        img_path = os.path.join(self.dataset_root, item['relative_path'])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.warning(f"Не вдалося завантажити: {img_path}. {e}")
            return torch.randn(3, 224, 224), torch.zeros(self.num_classes)

        if self.transform:
            image = self.transform(image)

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        ingredients = item['ingredients']
        for ing in ingredients:
            if ing in self.ingredient_to_index:
                index = self.ingredient_to_index[ing]
                target[index] = 1.0

        return image, target


# --- Модель ---
def get_model(num_classes):
    logging.info("Завантаження попередньо навченої моделі DenseNet-121")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model


# --- Функція навчання (Швидка версія) ---
def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs, patience):
    scaler = GradScaler()
    early_stopper = EarlyStopping(patience=patience, restore_best_weights=True)

    best_f1 = 0.0

    for epoch in range(num_epochs):
        logging.info(f"\n--- Епоха {epoch + 1}/{num_epochs} ---")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []

            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Епоха {epoch + 1}", unit="batch")

            for inputs, labels in progress_bar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(device_type=device.type, dtype=torch.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds)
                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)

            epoch_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
            epoch_precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)

            logging.info(
                f"{phase.capitalize()} Loss: {epoch_loss:.4f} | F1-micro: {epoch_f1:.4f} | Precision: {epoch_precision:.4f} | Recall: {epoch_recall:.4f}")

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    torch.save(model.state_dict(), f"../models/best_ingredient_model_f1_{best_f1:.4f}.pth")
                    logging.info(f"Збережено нову найкращу модель з F1-micro: {best_f1:.4f}")

                if early_stopper(epoch_loss, model):
                    model = early_stopper.restore_model(model)
                    logging.info("EarlyStopping: Тренування завершено достроково.")
                    return model

    early_stopper.restore_model(model)
    torch.save(model.state_dict(), "../models/final_model_best_loss.pth")
    return model


# --- Головна функція ---
def main():
    # !!! ЛОГУВАННЯ ПЕРЕНЕСЕНО СЮДИ ЩОБ НЕ БУЛО ПРОБЛЕМ З WORKERS !!!
    log_filename = f"../data/ingredient_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    logging.info("Скрипт тренування розпізнавання інгредієнтів запущено")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Використовується пристрій: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 2. Завантаження даних (приймає 4 значення)
    try:
        data_df, vocab, num_classes, pos_weight = load_and_prepare_data(
            FILE_RECIPES, FILE_IMAGE_ASSIGNMENTS, MIN_INGREDIENT_FREQUENCY
        )
    except Exception as e:
        logging.error(f"Помилка під час завантаження та обробки даних: {e}")
        return

    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)
    logging.info(f"Дані розділено: {len(train_df)} для тренування, {len(val_df)} для валідації.")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = IngredientDataset(
        train_df, vocab, num_classes, DATASET_ROOT_PATH, data_transforms['train']
    )
    val_dataset = IngredientDataset(
        val_df, vocab, num_classes, DATASET_ROOT_PATH, data_transforms['val']
    )

    dataloaders = {
        'train': DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2 if NUM_WORKERS > 0 else 2
        ),
        'val': DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2 if NUM_WORKERS > 0 else 2
        )
    }

    # 5. Ініціалізація
    model = get_model(num_classes).to(device)

    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    logging.info(f"Застосовано pos_weight до BCEWithLogitsLoss: {pos_weight:.2f}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    logging.info("Початок тренування (Single Phase)...")

    try:
        model = train_model(
            model, criterion, optimizer, scheduler, dataloaders,
            device, num_epochs=EPOCHS, patience=PATIENCE
        )
    except torch.cuda.OutOfMemoryError:
        logging.error("Помилка CUDA: не вистачає пам'яті. Спробуйте зменшити BATCH_SIZE.")
    except Exception as e:
        logging.error(f"Виникла помилка під час тренування: {e}")

    logging.info("Тренування завершено.")


if __name__ == "__main__":
    main()