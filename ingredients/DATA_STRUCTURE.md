# 📊 Структура даних для Ingredient CNN

Цей документ описує структуру всіх вхідних та вихідних файлів.

## 📥 Вхідні файли (обов'язкові)

### 1. `recipes_dataset_en_cleaned.json` (34 MB)

**Структура:** Dictionary з назвами рецептів як ключі

```json
{
  "apple_pie": {
    "recipe_name": "apple_pie",
    "category": "apple_pie",
    "ingredients": [
      "2 cups all-purpose flour",
      "1/2 cup granulated sugar",
      "1/4 tsp salt",
      "6 apples, peeled and sliced",
      "1/4 cup butter"
    ],
    "instructions": "Preheat oven to 350°F. Mix flour, sugar, and salt...",
    "source": "recipe1m"
  },
  "baby_back_ribs": { ... },
  ...
}
```

**Важні поля:**
- `recipe_name` - унікальна назва рецепта
- `ingredients` - список інгредієнтів у текстовому форматі
- `instructions` - інструкції приготування
- `category` - категорія їжі

**Кількість рецептів:**
- ~11k рецептів для 256 категорій Food-256
- Середня кількість: 40-50 рецептів на категорію

### 2. `image_to_recipe_assignments_f4_its.json` (22 MB)

**Структура:** Dictionary з шляхом до зображення як ключ

```json
{
  "images/apple_pie/00001.jpg": "apple_pie",
  "images/apple_pie/00002.jpg": "apple_pie",
  "images/baby_back_ribs/00001.jpg": "baby_back_ribs",
  ...
}
```

**Або альтернативна структура (список):**
```json
[
  {"image_path": "images/apple_pie/00001.jpg", "category": "apple_pie", "best_match_recipe_name": "apple_pie"},
  ...
]
```

**Розмір датасету:**
- ~150k зображень для Food-256
- ~600 зображень на категорію в середньому
- Розділено на train (80%) та valid (20%)

### 3. Датасет зображень (структура)

```
dataset_256/dataset/images/
├── apple_pie/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ... (600 зображень)
├── baby_back_ribs/
│   ├── 00001.jpg
│   └── ...
├── adobo/
├── almond_jelly/
└── ... (256 категорій всього)
```

**Параметри зображень:**
- **Формат:** JPEG
- **Розмір:** 224x224 пікселів (або буде ресайзено при завантаженні)
- **Колір:** RGB або будуть конвертовані у RGB
- **Всього:** ~150k зображень

---

## 📤 Вихідні файли (створюються скриптами)

### 1. `ingredient_vocabulary_V4_FINAL.json` (3 KB)

**Структура:** Dictionary інгредієнт → індекс

```json
{
  "salt": 0,
  "pepper": 1,
  "oil": 2,
  "butter": 3,
  "flour": 4,
  ...,
  "mascarpone": 153
}
```

**Властивості:**
- **154 інгредієнти** всього
- Індекси: 0-153 (мають відповідати вихідному шару моделі)
- Сортовано в порядку включення до словника
- Мінімум 20 зображень на інгредієнт

**Як створюється:**
```python
# У pytorch_gpu_universal_script.py
ing_to_idx = {}
for ing, count in sorted_ingredients_by_frequency:
    if count >= MIN_INGREDIENT_FREQUENCY:
        ing_to_idx[ing] = len(ing_to_idx)

# Результат: 154 інгредієнти
```

### 2. `best_ingredient_model_f1_0.4975.pth` (29 MB)

**Формат:** PyTorch model state_dict

```python
# Завантаження:
import torch

state_dict = torch.load('best_ingredient_model_f1_0.4975.pth')
model = models.densenet121()
model.classifier = nn.Linear(1024, 154)
model.load_state_dict(state_dict)
```

**Вміст:**
- Ваги DenseNet121 (ImageNet pre-trained + fine-tuned)
- Ваги Linear classifier (1024 → 154)

**Архітектура:**
```
DenseNet121
├── DenseNet layers (загальні)
│   ├── Conv + BatchNorm
│   ├── Dense blocks (1-4)
│   └── Output: (batch, 1024, 7, 7) після AdaptiveAvgPool
└── Classifier
    ├── Dropout(0.5)
    └── Linear(1024, 154)
```

### 3. `top_30_metrics.csv` (1.8 KB)

**Структура:** CSV з метриками для Top-30 інгредієнтів

```csv
Ingredient,Precision,Recall,F1-score
mascarpone,0.9230769230769231,0.75,0.8275862068965517
coconut oil,0.9473684210526315,0.5806451612903226,0.72
coffee,0.64,0.8,0.7111111111111111
croutons,0.7692307692307693,0.625,0.6896551724137931
...
```

**Колонки:**
- `Ingredient` - назва інгредієнта
- `Precision` - правильність детекції (true positives / predicted positives)
- `Recall` - повнота детекції (true positives / actual positives)
- `F1-score` - гармонійна середина Precision та Recall

**Сортування:** За F1-score від вищого до нижчого

**Кількість рядків:** 30 + header = 31 рядок

### 4. `top_30_ingredients_grouped.png`

**Формат:** PNG графік

**Вміст:**
- Коловий граф із трьома метриками для кожного інгредієнта
- Хороший для візуалізації качества моделі
- Розмір: 1920x1080 пікселів, 300 DPI

### 5. `evaluation_real_YYYYMMDD_HHMMSS.csv` (1-2 MB)

**Структура:** CSV з детальною evaluation (опціонально)

```csv
Category,Image_ID,Strategy,BLEU,ROUGE1,ROUGE2,ROUGEL,Cosine_Similarity,Generated_Text,Reference_Text,Prompt_Text
apple_pie,00001.jpg,Baseline,0.45,0.52,0.31,0.48,0.78,"Apple Pie\n\nIngredients:...",apple_pie\n\nIngredients:...,System: You are a Michelin-star...
apple_pie,00001.jpg,VisualContext,0.52,0.58,0.35,0.52,0.82,"Apple Pie with Vanilla...",apple_pie\n\nIngredients:...",System: You are a Michelin-star...
...
```

**Колонки:**
- `Category` - категорія зображення
- `Image_ID` - ім'я файлу
- `Strategy` - Baseline або VisualContext
- `BLEU` - метрика схожості тексту (BLEU)
- `ROUGE1/2/L` - метрики ROUGE
- `Cosine_Similarity` - схожість векторних представлень
- `Generated_Text` - згенерований рецепт від LLM
- `Reference_Text` - еталонний рецепт
- `Prompt_Text` - промпт для LLM

---

## 🔄 Трансформація даних (Ingredient Parsing)

### Вхід: Raw ingredient string
```
"2 cups all-purpose flour, sifted"
```

### Обробка:
1. **Видалення кількості та одиниць виміру:**
   ```
   "all purpose flour sifted"
   ```

2. **Видалення стоп-слів:**
   ```
   "all purpose flour"
   ```

3. **Пошук в CONSOLIDATION_MAP:**
   ```
   "all-purpose flour" → "flour"
   ```

4. **Консолідація:**
   ```
   "flour" ✓
   ```

### Вихід: Normalized ingredient
```
"flour"
```

### CONSOLIDATION_MAP приклади:

```python
{
    'all purpose flour': 'flour',
    'all-purpose flour': 'flour',
    'bread flour': 'flour',
    'cake flour': 'flour',

    'granulated sugar': 'sugar',
    'brown sugar': 'sugar',

    'olive oil': 'oil',
    'vegetable oil': 'oil',
    'canola oil': 'oil',

    'chicken breast': 'chicken',
    'chicken thigh': 'chicken',
    'ground chicken': 'chicken',

    # ... 150+ інших консолідацій
}
```

---

## 📊 Статистика

### Словник інгредієнтів

```
Всього 154 інгредієнти з MIN_INGREDIENT_FREQUENCY = 20

Розподіл за частотою:
- Top 10 інгредієнти: 30-50k зображень
- Top 30 інгредієнти: 10-30k зображень
- Решта 124 інгредієнти: 20-10k зображень
```

### Датасет

```
Категорії: 256
Зображення: ~150k
Рецепти: ~11k

Середня по категорії:
- Зображень: 586
- Рецептів: 43
- Унікальних інгредієнтів на рецепт: 5-8
```

---

## ✅ Валідація структури

### Перевірити рецепти:
```python
import json

with open('recipes_dataset_en_cleaned.json') as f:
    recipes = json.load(f)

print(f"Total recipes: {len(recipes)}")
print(f"Sample recipe: {list(recipes.keys())[0]}")
print(f"Sample fields: {list(recipes[list(recipes.keys())[0]].keys())}")
print(f"Sample ingredients: {recipes[list(recipes.keys())[0]]['ingredients'][:3]}")
```

### Перевірити зображення:
```python
import json

with open('image_to_recipe_assignments_f4_its.json') as f:
    assignments = json.load(f)

print(f"Total images: {len(assignments)}")
print(f"Sample mapping: {list(assignments.items())[0]}")

# Перевірити категорії
categories = set(assignments.values())
print(f"Unique categories: {len(categories)}")
```

### Перевірити словник:
```python
import json

with open('ingredient_vocabulary_V4_FINAL.json') as f:
    vocab = json.load(f)

print(f"Total ingredients: {len(vocab)}")
print(f"Indices range: {min(vocab.values())}-{max(vocab.values())}")
print(f"Sample: {list(vocab.items())[:5]}")
```

---

**Документація структури даних | 2025**
