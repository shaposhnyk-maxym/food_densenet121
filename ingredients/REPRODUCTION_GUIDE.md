# 🔄 Гайд для повного відтворення результатів Ingredient CNN

## 📋 Передумови

Цей гайд описує всі кроки для відтворення результатів детекції інгредієнтів з файлу `top_30_metrics.csv`.

## ✅ Вимоги

### Python залежності:
```
torch>=2.0
torchvision>=0.15
scikit-learn>=1.0
pandas>=1.3
numpy>=1.20
tqdm
Pillow
matplotlib
seaborn
```

### Hardware:
- **Рекомендовано**: GPU (NVIDIA CUDA)
- **Мінімум**: 8 GB RAM
- **Місце на диску**: ~50 GB (для датасету)

### Опціонально (для повної evaluation):
- Ollama з моделлю `llama3.1`
- nltk, rouge-score

## 📁 Організація файлів

Перед початком переконайтеся, що у вас є:

```
ingredients/
├── data/
│   ├── recipes_dataset_en_cleaned.json           (34 MB)
│   ├── image_to_recipe_assignments_f4_its.json   (22 MB)
│   └── [datasets]/
│       └── dataset_256/images/
│           ├── apple_pie/
│           ├── baby_back_ribs/
│           └── ... (256 категорій)
└── scripts/
    ├── pytorch_gpu_universal_script.py
    ├── eval_per_class_grouped.py
    └── run_pipeline_real.py
```

## 🚀 Крок 1: Тренування моделі

### Команда:
```bash
cd scripts
python pytorch_gpu_universal_script.py
```

### Що відбувається:

1. **Завантаження даних** (2-3 хв)
   - Читає рецепти з JSON
   - Завантажує зображення та інгредієнти
   - Парсує та консолідує інгредієнти

2. **Створення словника** (~1 хв)
   - Вибирає інгредієнти з мін. 20 зображеннями
   - Створює `ingredient_vocabulary_V4_FINAL.json` (154 інгредієнти)

3. **Тренування** (1-2 години на GPU)
   - 50 епох з Early Stopping
   - Batch size: 32
   - Learning rate: 1e-4 (з decay)

4. **Збереження результатів:**
   - `best_ingredient_model_f1_0.4975.pth` - модель
   - `ingredient_vocabulary_V4_FINAL.json` - словник
   - `ingredient_training_*.log` - логи

### Параметри конфігурації (в скрипті):

```python
FILE_RECIPES = 'recipes_dataset_en_cleaned.json'
FILE_IMAGE_ASSIGNMENTS = 'image_to_recipe_assignments_f4_its.json'
DATASET_ROOT_PATH = '../'
MIN_INGREDIENT_FREQUENCY = 20  # Мінімум зображень на інгредієнт
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10  # Early stopping
NUM_WORKERS = 4
DEVICE = 'cuda'  # Автоматично обирає GPU
```

### Очікувані результати:

```
Loaded 154 ingredients (Should be 154).
Training epoch 1/50: Loss=0.324, Val F1=0.421
...
Training epoch 50/50: Loss=0.087, Val F1=0.4975
Model saved: best_ingredient_model_f1_0.4975.pth
```

## 🚀 Крок 2: Генерація Top-30 метрик

### Команда:
```bash
cd scripts
python eval_per_class_grouped.py
```

### Що створюється:

1. **Вихідні файли:**
   - `top_30_metrics.csv` - основний результат
   - `top_30_ingredients_grouped.png` - графік метрик

2. **Структура CSV:**
   ```
   Ingredient,Precision,Recall,F1-score
   mascarpone,0.923,0.75,0.827
   coconut oil,0.947,0.581,0.72
   ...
   breadcrumbs,0.42,0.631,0.504
   ```

### Очікувана помилка чи проблема?

**Помилка:** `FileNotFoundError: ingredient_vocabulary_V4_FINAL.json`
- **Рішення:** Спочатку запустіть Крок 1 (тренування)

**Помилка:** `Model size mismatch`
- **Рішення:** Перевірте що `num_classes = 154` у скрипті

## 🚀 Крок 3: Повна evaluation pipeline (опціонально)

### Вимоги:
- Ollama встановлена та запущена
- Модель `llama3.1` завантажена (`ollama pull llama3.1`)

### Команда:
```bash
cd scripts
python run_pipeline_real.py
```

### Що робить:

1. Навантажує модель та словник
2. Для кожної категорії (20 категорій):
   - Вибирає 5 рандомних зображень
   - Детектує інгредієнти DenseNet121
   - Генерує 2 рецепти via LLM (Baseline + Visual Context)
   - Обраховує BLEU, ROUGE, Cosine Similarity метрики

3. Зберігає результати в `evaluation_real_YYYYMMDD_HHMMSS.csv`

### Конфігурація:
```python
TOP_N_CATEGORIES = 20
SAMPLES_PER_CATEGORY = 5
CONFIDENCE_THRESHOLD = 0.15
MODEL_LLM = "llama3.1"
```

### Очікуваний час: 30-60 хвилин

## 📊 Результати та метрики

### Що шукати в `top_30_metrics.csv`:

| Метрика | Опис |
|---------|------|
| **Precision** | Яка частка детектованих інгредієнтів дійсно присутня |
| **Recall** | Яка частка справжніх інгредієнтів була детектована |
| **F1-score** | Гармонійна середина Precision та Recall |

### Інтерпретація результатів:

```
F1 >= 0.80: Відмінно (mascarpone, coconut oil)
F1 >= 0.70: Добре (coffee, croutons)
F1 >= 0.60: Задовільно (lamb, bbq sauce)
F1 >= 0.50: Прийнятно (більша частина)
```

## 🔧 Налагодження та оптимізація

### Якщо F1-score низький (< 0.40):

1. **Перевірити дані:**
   ```bash
   python -c "import json; d=json.load(open('data/recipes_dataset_en_cleaned.json')); print(f'Recipes: {len(d)}')"
   ```

2. **Збільшити епохи:**
   - Змініть `EPOCHS = 100` у скрипті

3. **Налаштувати learning rate:**
   - Спробуйте `LEARNING_RATE = 5e-5` або `1e-3`

4. **Перевірити словник:**
   ```bash
   python -c "import json; v=json.load(open('models/ingredient_vocabulary_V4_FINAL.json')); print(f'Classes: {len(v)}')"
   ```

### Якщо модель перекомплект (Recall низький):

- Збільшити Dropout: `nn.Dropout(0.7)` замість `0.5`
- Додати регуляризацію L2
- Зменшити `MIN_INGREDIENT_FREQUENCY` на 10-15

### Якщо модель недокомплект (Precision низький):

- Зменшити Dropout: `nn.Dropout(0.3)`
- Збільшити Learning rate
- Тренувати більше епох

## 🧪 Верифікація результатів

### 1. Перевірити файли:
```bash
# Словник має 154 інгредієнти
python -c "import json; print(len(json.load(open('models/ingredient_vocabulary_V4_FINAL.json'))))"

# Модель має 29 MB
ls -lh models/best_ingredient_model_f1_0.4975.pth

# CSV має 30 рядків (31 з header)
wc -l top_30_metrics.csv
```

### 2. Перевірити метрики:
```python
import pandas as pd

df = pd.read_csv('top_30_metrics.csv')
print(f"Mean F1: {df['F1-score'].mean():.4f}")
print(f"Max F1: {df['F1-score'].max():.4f}")
print(f"Min F1: {df['F1-score'].min():.4f}")
```

### 3. Тест на новому зображенні:
```bash
python -c "
import torch
from torchvision import transforms, models
from PIL import Image
import json

# Завантажити модель
model = models.densenet121()
model.classifier = torch.nn.Linear(1024, 154)
model.load_state_dict(torch.load('models/best_ingredient_model_f1_0.4975.pth'))
model.eval()

# Словник
with open('models/ingredient_vocabulary_V4_FINAL.json') as f:
    ing_to_idx = json.load(f)
vocab = [None] * len(ing_to_idx)
for ing, idx in ing_to_idx.items():
    vocab[idx] = ing

# Зображення
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open('test_image.jpg').convert('RGB')
with torch.no_grad():
    out = model(transform(img).unsqueeze(0))
    probs = torch.sigmoid(out).squeeze(0).numpy()

detected = [(vocab[i], float(p)) for i, p in enumerate(probs) if p > 0.5]
for ing, conf in sorted(detected, key=lambda x: -x[1])[:10]:
    print(f'{ing}: {conf:.2%}')
"
```

## 📈 Порівняння результатів

Якщо ви повторили трейнінг, порівняйте з оригіналом:

```python
import pandas as pd

original = pd.read_csv('reference_top_30_metrics.csv')
new = pd.read_csv('top_30_metrics.csv')

print(f"Original Mean F1: {original['F1-score'].mean():.4f}")
print(f"New Mean F1: {new['F1-score'].mean():.4f}")
```

Невеликі різниці (±0.02) є нормальними через випадковість у тренуванні.

## ⏱️ Очікуваний час виконання

| Крок | Час |
|------|-----|
| Завантаження даних | 2-3 хв |
| Тренування (50 епох) | 1-2 години (GPU) / 8+ годин (CPU) |
| Генерація метрик | 10-15 хв |
| Повна evaluation | 30-60 хв |
| **Всього** | **2-3 години (GPU)** |

## 🐛 Вирішення проблем

### Problem: CUDA out of memory
```
Solution: Зменшити BATCH_SIZE на 16 або 8
```

### Problem: Модель не покращується
```
Solution:
1. Перевірити дані (чи достатньо примірів)
2. Збільшити LEARNING_RATE в 2 рази
3. Видалити Early Stopping (PATIENCE = 50)
```

### Problem: Дуже висока loss
```
Solution:
1. Скоротити LEARNING_RATE в 10 разів
2. Перевірити що дані завантажуються правильно
3. Перевірити device (cuda vs cpu)
```

## 📚 Додаткові ресурси

- `README.md` - Огляд модуля
- `scripts/pytorch_gpu_universal_script.py` - Повний код тренування
- `scripts/eval_per_class_grouped.py` - Код для метрик

---

**Гайд для повного відтворення | 2025**
