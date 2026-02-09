# 🥘 Ingredient Detection Pipeline

Автоматична детекція інгредієнтів на зображеннях їжі за допомогою DenseNet121 CNN моделі.

## 📋 Огляд

Цей модуль розпізнає **154 різних інгредієнти** на зображеннях їжі, використовуючи мультилейбл класифікацію. Модель навчена на датасеті Food-101/Food-256 з відповідністю до рецептів та їх інгредієнтів.

## 🏆 Результати

**Top-30 інгредієнтів (за F1-score):**

| Ранг | Інгредієнт | Precision | Recall | F1-Score |
|------|-----------|-----------|--------|----------|
| 1 | mascarpone | 0.923 | 0.750 | 0.827 |
| 2 | coconut oil | 0.947 | 0.581 | 0.720 |
| 3 | coffee | 0.640 | 0.800 | 0.711 |
| 4 | croutons | 0.769 | 0.625 | 0.690 |
| 5 | lamb | 0.600 | 0.733 | 0.660 |
| ... | ... | ... | ... | ... |
| 30 | breadcrumbs | 0.420 | 0.631 | 0.504 |

**Повний набір метрик:** див. `data/top_30_metrics.csv`

## 🔧 Архітектура моделі

```python
DenseNet121 (ImageNet pre-trained)
├── Conv Features (ImageNet weights)
├── Adaptive Pool
└── Classifier
    ├── Dropout(0.5)
    └── Linear(1024 → 154)  # Multi-label outputs
```

- **Базова модель**: DenseNet121 з ImageNet weights
- **Вихідні класи**: 154 інгредієнти
- **Активація**: Sigmoid (для Multi-label)
- **Порог передбачення**: 0.5

## 📁 Структура

```
ingredients/
├── README.md                              # Цей файл
├── REPRODUCTION_GUIDE.md                  # Повний гайд для відтворення
├── models/
│   ├── best_ingredient_model_f1_0.4975.pth    # Навчена модель
│   └── ingredient_vocabulary_V4_FINAL.json    # Словник (154 інгредієнти)
├── scripts/
│   ├── pytorch_gpu_universal_script.py         # Тренування моделі
│   ├── eval_per_class_grouped.py               # Генерація Top-30 метрик
│   └── run_pipeline_real.py                    # Повна evaluation pipeline
└── data/
    ├── top_30_metrics.csv                      # Top-30 результати
    ├── recipes_dataset_en_cleaned.json         # Датасет рецептів (34 MB)
    └── image_to_recipe_assignments_f4_its.json # Зображення↔Рецепти (22 MB)
```

## 🚀 Швидкий старт

### 1️⃣ Встановлення залежностей

```bash
pip install torch torchvision scikit-learn pandas numpy tqdm
```

### 2️⃣ Завантажити модель та словник

Використовуйте файли з папки `models/`:
- `best_ingredient_model_f1_0.4975.pth` - модель (29 MB)
- `ingredient_vocabulary_V4_FINAL.json` - словник інгредієнтів

### 3️⃣ Зробити передбачення на зображенні

```python
import torch
from torchvision import transforms, models
import json
from PIL import Image

# Завантажити модель
model = models.densenet121()
model.classifier = torch.nn.Linear(1024, 154)
model.load_state_dict(torch.load('models/best_ingredient_model_f1_0.4975.pth'))
model.eval()

# Завантажити словник
with open('models/ingredient_vocabulary_V4_FINAL.json') as f:
    ing_to_idx = json.load(f)
    vocab = [None] * len(ing_to_idx)
    for ing, idx in ing_to_idx.items():
        vocab[idx] = ing

# Подготовити зображення
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Передбачення
image = Image.open('food_image.jpg').convert('RGB')
with torch.no_grad():
    output = model(transform(image).unsqueeze(0))
    probs = torch.sigmoid(output).squeeze(0).numpy()

# Результати (порог 0.5)
detected = [(vocab[i], float(p)) for i, p in enumerate(probs) if p > 0.5]
detected.sort(key=lambda x: x[1], reverse=True)
print(detected)
```

## 📊 Тренування моделі

Див. `REPRODUCTION_GUIDE.md` для детального гайду.

### Швидкий запуск:

```bash
cd scripts
python pytorch_gpu_universal_script.py
```

**Параметри:**
- `BATCH_SIZE = 32`
- `EPOCHS = 50`
- `LEARNING_RATE = 1e-4`
- `MIN_INGREDIENT_FREQUENCY = 20`

## 📈 Evaluation

### Генерація Top-30 метрик:

```bash
cd scripts
python eval_per_class_grouped.py
```

Результат: `top_30_metrics.csv` + `top_30_ingredients_grouped.png`

### Повна evaluation pipeline:

```bash
cd scripts
python run_pipeline_real.py
```

Потребує:
- Ollama з моделлю `llama3.1`
- Генерує детальні метрики BLEU, ROUGE, Cosine Similarity

## 📦 Вхідні дані для навчання

**Три обов'язкові JSON файли:**

1. **recipes_dataset_en_cleaned.json** (34 MB)
   - Структура: `{ recipe_name: { ingredients: [...], instructions: "...", ... }, ... }`
   - 11k+ рецептів для Food-256 категорій

2. **image_to_recipe_assignments_f4_its.json** (22 MB)
   - Структура: `{ image_path: category, ... }`
   - Відповідність зображень до категорій рецептів

3. **Датасет зображень** (Food-256)
   - Структура: `dataset_256/dataset/images/{category}/{image}.jpg`
   - 256 категорій, ~150k зображень

**Автоматично створюється під час навчання:**
- `ingredient_vocabulary_V4_FINAL.json` - словник з 154 інгредієнтів

## 🔍 Парсинг інгредієнтів

Інгредієнти проходять нормалізацію та консолідацію:

```
"olive oil" → "oil"
"granulated sugar" → "sugar"
"chicken breast" → "chicken"
```

- Видаляються стоп-слова та одиниці виміру
- Мінімум 20 зображень на інгредієнт для включення у навчання
- 154 остаточних класів

## ⚠️ Важливі моменти

1. **Порядок словника критичний** — індекс має відповідати нейронам вихідного шару
2. **Структура шляхів** — скрипти очікують датасет на `../dataset_256/`
3. **GPU рекомендується** — навчання на CPU буде повільним
4. **Мультилейбл класифікація** — одне зображення може містити багато інгредієнтів

## 🧪 Тестування

### Перевірити дані:

```bash
python -c "import json; v=json.load(open('models/ingredient_vocabulary_V4_FINAL.json')); print(f'Vocabulary: {len(v)} ingredients')"
python -c "import json; r=json.load(open('data/recipes_dataset_en_cleaned.json')); print(f'Recipes: {len(r)}')"
```

### Перевірити модель:

```bash
python -c "import torch; m=torch.load('models/best_ingredient_model_f1_0.4975.pth'); print(f'Model size: {sum(p.numel() for p in m.values())} params')"
```

## 📚 Посилання

- [DenseNet Paper](https://arxiv.org/abs/1608.06993)
- [Food-101 Dataset](https://www.tensorflow.org/datasets/catalog/food101)
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)

## 📝 Ліцензія

Див. основний репозиторій

---

**Інгредієнтна детекція | DenseNet121 | 154 класи | 2025**
