import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

# --- НАЛАШТУВАННЯ ---
# Шляхи адаптовані для структури репозиторію food_densenet_121
FILE_RECIPES = '../data/recipes_dataset_en_cleaned.json'
FILE_IMAGE_ASSIGNMENTS = '../data/image_to_recipe_assignments_f4_its.json'
VOCAB_FILE = '../models/ingredient_vocabulary_V4_FINAL.json'  # <--- ОБОВ'ЯЗКОВО ЦЕЙ ФАЙЛ
DATASET_ROOT_PATH = '../../datasets/food256/'
MODEL_PATH = '../models/best_ingredient_model_f1_0.4975.pth'
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- СТОП-СЛОВА І MAP (для парсингу вхідних даних) ---
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
SORTED_CONSOLIDATION_KEYS = sorted(CONSOLIDATION_MAP.keys(), key=len, reverse=True)


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


class IngredientDataset(Dataset):
    def __init__(self, recipes_file, assignments_file, vocab_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        print("Loading data...")
        with open(recipes_file, 'r', encoding='utf-8') as f:
            raw_recipes = json.load(f)
        with open(assignments_file, 'r', encoding='utf-8') as f:
            raw_assignments = json.load(f)

        # --- ВАЖЛИВО: Завантажуємо оригінальний словник ---
        print(f"Loading vocabulary from {vocab_file}...")
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(
                f"НЕ ЗНАЙДЕНО ФАЙЛ: {vocab_file}. Будь ласка, знайдіть файл ingredient_vocabulary_V4_FINAL.json, який був створений під час тренування!")

        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.ing_to_idx = json.load(f)

        # Створюємо список слів, де index 0 це слово 0, index 1 це слово 1...
        # Це критично, щоб порядок відповідав вагам моделі
        self.vocab = [None] * len(self.ing_to_idx)
        for ing, idx in self.ing_to_idx.items():
            self.vocab[idx] = ing

        print(f"Loaded {len(self.vocab)} ingredients (Should be 154).")

        # Fix Recipes
        if isinstance(raw_recipes, list):
            sample = raw_recipes[0]
            title_key = next((k for k in sample.keys() if k in ['recipe_name', 'title', 'name']), None)
            self.recipes = {r[title_key]: r for r in raw_recipes}
        else:
            self.recipes = raw_recipes

        # Fix Assignments
        if isinstance(raw_assignments, list):
            sample = raw_assignments[0]
            if isinstance(sample, dict):
                img_key = next((k for k in sample.keys() if 'img' in k or 'path' in k or 'image' in k), None)
                rec_key = next((k for k in sample.keys() if 'recipe' in k or 'title' in k or 'class' in k), None)
                self.assignments = {x[img_key]: x[rec_key] for x in raw_assignments}
            elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                self.assignments = {x[0]: x[1] for x in raw_assignments}
        else:
            self.assignments = raw_assignments

        # Filter
        self.samples = []
        print("Filtering samples using loaded vocabulary...")
        for img_path, recipe_title in tqdm(self.assignments.items()):
            if recipe_title in self.recipes:
                recipe = self.recipes[recipe_title]
                has_valid = False
                for ing in recipe.get('ingredients', []):
                    parsed = parse_ingredient(ing)
                    if parsed and parsed in self.ing_to_idx:
                        has_valid = True
                        break
                if has_valid:
                    self.samples.append((img_path, recipe_title))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, recipe_title = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_rel_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        recipe = self.recipes[recipe_title]
        label = torch.zeros(len(self.vocab), dtype=torch.float32)

        for ing in recipe.get('ingredients', []):
            parsed = parse_ingredient(ing)
            if parsed and parsed in self.ing_to_idx:
                label[self.ing_to_idx[parsed]] = 1.0

        return image, label


def get_model(num_classes):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    # --- ЗМІНА ТУТ: Додали Dropout ---
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    # ---------------------------------
    return model


def evaluate_and_plot_grouped():
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        dataset = IngredientDataset(FILE_RECIPES, FILE_IMAGE_ASSIGNMENTS, VOCAB_FILE, DATASET_ROOT_PATH,
                                    transform=val_transform)
    except FileNotFoundError as e:
        print(f"\n!!! ПОМИЛКА: {e}")
        return

    # Validation subset
    indices = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.2 * len(dataset))
    val_indices = indices[:split]

    from torch.utils.data import Subset
    val_dataset = Subset(dataset, val_indices)
    dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    num_classes = len(dataset.vocab)
    print(f"Initializing model with {num_classes} classes...")
    model = get_model(num_classes)

    print("Loading weights...")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_targets = []

    print("Inference...")
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None,
                                                                     zero_division=0)

    metrics_df = pd.DataFrame({
        'Ingredient': dataset.vocab,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    })

    metrics_df.sort_values(by='F1-score', ascending=False, inplace=True)
    top_30 = metrics_df.head(30)
    top_30.to_csv('../data/top_30_metrics.csv', index=False)

    top_30_melted = top_30.melt(id_vars='Ingredient', var_name='Metric', value_name='Score')

    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")

    sns.barplot(
        data=top_30_melted,
        x='Ingredient',
        y='Score',
        hue='Metric',
        palette={'Precision': '#2ca02c', 'Recall': '#d62728', 'F1-score': '#1f77b4'}
    )

    plt.title('Top 30 Ingredients Performance Metrics', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.05)
    plt.legend(title='Metric', loc='upper right')
    plt.tight_layout()

    plt.savefig('../data/top_30_ingredients_grouped.png', dpi=300)
    print("SUCCESS! top_30_ingredients_grouped.png created in ../data/")


if __name__ == '__main__':
    evaluate_and_plot_grouped()