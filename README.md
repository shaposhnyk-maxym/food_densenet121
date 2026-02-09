# 🍽️ Food DenseNet121 - Класифікація їжі та детекція інгредієнтів

Репозиторій з двома основними моделями на основі DenseNet121:
1. **🍕 Категоризація їжі** - класифікація 256 категорій
2. **🥘 Детекція інгредієнтів** - розпізнавання 154 інгредієнтів

## 📁 Структура проекту

```
food_densenet_121/
├── README.md                           # Цей файл
├── categories/                         # 🍕 Модель категоризації
│   ├── pytorch_gpu_universal_trainer.py    # Тренування
│   ├── evaluate_finetuned_model.py         # Оцінювання
│   └── results/                        # Результати
│       ├── evaluation_results.csv
│       └── confusion_matrix.csv
├── datasets/
│   └── food256/                        # Food 256 датасет
│       ├── train/
│       └── valid/
├── ingredients/                        # 🥘 Модель детекції інгредієнтів
│   ├── README.md                       # Детальна документація
│   ├── REPRODUCTION_GUIDE.md           # Гайд для відтворення
│   ├── DATA_STRUCTURE.md               # Структура даних
│   ├── models/
│   │   ├── best_ingredient_model_f1_0.4975.pth
│   │   └── ingredient_vocabulary_V4_FINAL.json
│   ├── scripts/
│   │   ├── pytorch_gpu_universal_script.py
│   │   ├── eval_per_class_grouped.py
│   │   └── run_pipeline_real.py
│   └── data/
│       ├── top_30_metrics.csv
│       ├── recipes_dataset_en_cleaned.json
│       └── image_to_recipe_assignments_f4_its.json
├── models/                             # Папка для моделей
│   └── food_256_finetuned.h5          # Category model
└── outputs/                            # Вихідні файли
```

## 🚀 Швидкий старт

### 🍕 Категоризація їжі (256 категорій)

#### 1️⃣ Підготовка датасету

Розпакуйте Food 256 датасет в папку `datasets/food256/`:
```
datasets/food256/
├── train/
│   ├── adobo/
│   ├── almond_jelly/
│   └── ...
└── valid/
    ├── adobo/
    ├── almond_jelly/
    └── ...
```

#### 2️⃣ Тренування моделі

```bash
cd categories
python pytorch_gpu_universal_trainer.py
```

#### 3️⃣ Оцінювання

```bash
cd categories
python evaluate_finetuned_model.py
```

**Результати:**
- Precision: 95.47%
- Recall: 94.60%
- F1 Score: 94.95%
- Accuracy: 92.59%

---

### 🥘 Детекція інгредієнтів (154 інгредієнти)

Автоматична детекція інгредієнтів на зображеннях їжі.

#### 📊 Результати Top-30 інгредієнтів:

```
Mascarpone:      F1=0.827 (Precision=0.923, Recall=0.75)
Coconut oil:     F1=0.720 (Precision=0.947, Recall=0.58)
Coffee:          F1=0.711 (Precision=0.640, Recall=0.80)
Croutons:        F1=0.690 (Precision=0.769, Recall=0.63)
Lamb:            F1=0.660 (Precision=0.600, Recall=0.73)
...
```

**Див. повний список:** `ingredients/data/top_30_metrics.csv`

#### 🚀 Запуск детекції інгредієнтів

**Див. детальну документацію:** `ingredients/README.md`

**Швидкий старт:**

```python
import torch
from torchvision import transforms, models
from PIL import Image
import json

# Завантажити модель
model = models.densenet121()
model.classifier = torch.nn.Linear(1024, 154)
model.load_state_dict(torch.load('ingredients/models/best_ingredient_model_f1_0.4975.pth'))
model.eval()

# Словник інгредієнтів
with open('ingredients/models/ingredient_vocabulary_V4_FINAL.json') as f:
    ing_to_idx = json.load(f)
    vocab = [None] * len(ing_to_idx)
    for ing, idx in ing_to_idx.items():
        vocab[idx] = ing

# Передбачення
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('food_image.jpg').convert('RGB')
with torch.no_grad():
    output = model(transform(image).unsqueeze(0))
    probs = torch.sigmoid(output).squeeze(0).numpy()

# Результати
detected = [(vocab[i], float(p)) for i, p in enumerate(probs) if p > 0.5]
detected.sort(key=lambda x: x[1], reverse=True)
for ingredient, confidence in detected[:5]:
    print(f"{ingredient}: {confidence:.1%}")
```

#### 📚 Документація

- **README.md** - Огляд, архітектура, швидкий старт
- **REPRODUCTION_GUIDE.md** - Повний гайд для тренування та оцінювання
- **DATA_STRUCTURE.md** - Структура всіх вхідних/вихідних файлів

## 📊 Результати

### 🍕 Категоризація (256 класів)

## 🛠️ Вимоги

### Основні
```
torch>=2.0
torchvision>=0.15
scikit-learn>=1.0
pandas>=1.3
matplotlib>=3.5
seaborn>=0.11
numpy>=1.20
Pillow
tqdm
```

### Опціонально (для інгредієнтів)
```
ollama                # Для LLM-based evaluation
nltk                 # Для текстових метрик
rouge-score          # Для ROUGE метрик
```

## 🖥️ Системні вимоги

- **GPU:** Рекомендується NVIDIA CUDA (для швидкого тренування)
- **RAM:** Мінімум 8 GB
- **Місце:** 50+ GB для датасету

## 📥 Дані (Ingredient Pipeline)

⚠️ **Великі файли даних НЕ включені в репозиторій!**

Для повного тренування інгредієнтної моделі потрібні:

1. `recipes_dataset_en_cleaned.json` (34 MB) - Рецепти
2. `image_to_recipe_assignments_f4_its.json` (22 MB) - Зображення↔Рецепти
3. `dataset_256/images/` (~150k зображень) - Датасет Food-256

**Де взяти дані:**
- Рецепти: Recipe1M датасет або ваша колекція
- Зображення: Food-101/Food-256 датасет
- Див. `ingredients/DATA_STRUCTURE.md` для структури

**Налаштування шляхів:**

Відредагуйте шляхи в скриптах `ingredients/scripts/`:

```python
FILE_RECIPES = 'path/to/recipes_dataset_en_cleaned.json'
FILE_IMAGE_ASSIGNMENTS = 'path/to/image_to_recipe_assignments_f4_its.json'
DATASET_ROOT_PATH = 'path/to/dataset_256/'
```

## 📦 Файли в репозиторії

### Включено ✅
- ✅ Скрипти для тренування та оцінювання
- ✅ Архітектури моделей
- ✅ Конфіги та параметри
- ✅ Документація та гайди

### НЕ включено (занадто великі) ❌
- ❌ Датасет Food-256 (150k зображень)
- ❌ Рецепти JSON (34 MB)
- ❌ Навчені моделі (29 MB .pth файли)
- ❌ Вихідні CSV файли з результатами

## 🔍 Що нового порівняно з категоризацією?

| Функція | Категорізація | Інгредієнти |
|---------|--|--|
| Модель | DenseNet121 | DenseNet121 |
| Класи | 256 (категорії) | 154 (інгредієнти) |
| Тип завдання | Single-label | Multi-label |
| Точність (F1) | 0.95 | 0.50* |
| Top результат | 95% | Mascarpone: 0.83 |
| Вхід | 1 зображення | 1 зображення |
| Вихід | 1 категорія | Список інгредієнтів |

*Top-30 середня F1-score для інгредієнтів

## 🧪 Тестування

### Категоризація
```bash
cd categories
python evaluate_finetuned_model.py
```

### Інгредієнти
```bash
cd ingredients/scripts
python eval_per_class_grouped.py
```

## 📚 Документація

### 🍕 Категоризація
- Базові скрипти в папці `categories/`
- README в корені репозиторію

### 🥘 Інгредієнти
- `ingredients/README.md` - Огляд та архітектура
- `ingredients/REPRODUCTION_GUIDE.md` - Крок за кроком гайд
- `ingredients/DATA_STRUCTURE.md` - Формат даних
- `ingredients/scripts/` - Всі скрипти

## 🔗 Посилання

**Репозиторій:**
```
https://github.com/shaposhnyk-maxym/food_densenet121.git
```

**Датасети:**
- [Food-101](https://www.tensorflow.org/datasets/catalog/food101)
- [Recipe1M](http://pic2recipe.csail.mit.edu/)

**Модельні архітектури:**
- [DenseNet Paper](https://arxiv.org/abs/1608.06993)
- [PyTorch DenseNet](https://pytorch.org/vision/stable/models.html#densenet)

## 📝 Ліцензія

MIT License

---

**DenseNet121 | Категоризація + Інгредієнти | Food-256 | 2025**
