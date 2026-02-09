import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
import csv
import random
import time
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import ollama
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# --- АВТОМАТИЧНЕ ЗАВАНТАЖЕННЯ NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

# ==========================================
# КОНФІГУРАЦІЯ
# ==========================================
# Файли даних - адаптовані для структури репозиторію food_densenet_121
FILE_RECIPES = '../data/recipes_dataset_en_cleaned.json'
FILE_ASSIGNMENTS = '../data/image_to_recipe_assignments_f4_its.json'
VOCAB_FILE = '../models/ingredient_vocabulary_V4_FINAL.json'
MODEL_PATH = '../models/best_ingredient_model_f1_0.4975.pth'

# Шлях до кореня датасету Food-256 (відносно скрипта)
DATASET_ROOT_PATH = '../../datasets/food256/'

# Налаштування LLM
MODEL_LLM = "llama3.1"  # Переконайся, що ця модель є в 'ollama list'

# Налаштування експерименту
TOP_N_CATEGORIES = 20  # Кількість категорій для тесту
SAMPLES_PER_CATEGORY = 5  # Кількість рецептів на кожну категорію (постав 1-2 для швидкого тесту, 5-10 для звіту)
CONFIDENCE_THRESHOLD = 0.15  # Поріг DenseNet (беремо все, що > 10%, щоб дати Лламі вибір)

# Пристрій
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

# Ім'я вихідного файлу (зберігається в папку data)
OUTPUT_CSV = f'../data/evaluation_real_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'


# ==========================================
# 1. МОДЕЛЬ DENSENET (PYTORCH)
# ==========================================
def get_densenet_model(num_classes):
    """Створює архітектуру DenseNet121 під нашу кількість класів."""
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model


def load_inference_components():
    """Завантажує словник інгредієнтів та ваги моделі."""
    if not os.path.exists(VOCAB_FILE):
        raise FileNotFoundError(f"Vocab file not found: {VOCAB_FILE}")

    print(f"Loading vocabulary from {VOCAB_FILE}...")
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        ing_to_idx = json.load(f)

    # Створюємо список, де індекс відповідає нейрону вихідного шару
    vocab = [None] * len(ing_to_idx)
    for ing, idx in ing_to_idx.items():
        vocab[idx] = ing

    num_classes = len(vocab)
    print(f"Vocabulary size: {num_classes} ingredients")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    print(f"Loading model weights from {MODEL_PATH}...")
    model = get_densenet_model(num_classes)

    # Завантаження ваг (map_location важливий, якщо тренували на GPU, а запускаємо на CPU)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    new_state_dict = {}
    for k, v in state_dict.items():
        # Прибираємо зайву ".1" з назви ключів класифікатора
        if k.startswith('classifier.1.'):
            new_key = k.replace('classifier.1.', 'classifier.')
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # 3. Завантажуємо виправлений словник
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()

    # Трансформації для інференсу (стандартні ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return model, vocab, transform


def predict_ingredients(model, vocab, transform, img_path):
    """Проганяє зображення через модель і повертає список (інгредієнт, впевненість)."""
    if not os.path.exists(img_path):
        # Спробуємо знайти файл, ігноруючи абсолютний шлях з JSON
        filename = os.path.basename(img_path)
        # Припускаємо структуру food101/dataset/images/category/filename.jpg
        # Спробуємо знайти категорію в шляху
        parts = img_path.replace('\\', '/').split('/')
        if 'images' in parts:
            idx = parts.index('images')
            if idx + 2 < len(parts):
                cat = parts[idx + 1]
                name = parts[idx + 2]
                alt_path = os.path.join(DATASET_ROOT_PATH, 'dataset_256', 'dataset', 'images', cat, name)
                if os.path.exists(alt_path):
                    img_path = alt_path
                else:
                    print(f"Warning: Image not found at {img_path} or {alt_path}")
                    return []
        else:
            print(f"Warning: Image file not found: {img_path}")
            return []

    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return []

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        # Sigmoid для Multi-label
        probs = torch.sigmoid(output).squeeze(0).cpu().numpy()

    detected = []
    for idx, conf in enumerate(probs):
        if conf > CONFIDENCE_THRESHOLD:
            detected.append((vocab[idx], float(conf)))

    # Сортуємо: найбільш вірогідні зверху
    detected.sort(key=lambda x: x[1], reverse=True)
    return detected


# ==========================================
# 2. УТИЛІТИ ТЕКСТУ ТА МЕТРИК
# ==========================================
def format_ground_truth(name, ingredients, instructions):
    """
    Формує еталонний текст рецепта.
    Важливо: формат має бути ідентичним тому, що ми просимо у LLM.
    """
    # Очистка інструкцій від зайвих пробілів
    clean_instr = re.sub(r'\s+', ' ', instructions).strip()

    # Формуємо список інгредієнтів
    ing_str = "\n".join([f"{i + 1}. {ing}" for i, ing in enumerate(ingredients)])

    return f"{name}\n\nIngredients:\n{ing_str}\n\nInstructions:\n{clean_instr}"


def get_embedding(text):
    """Отримує ембеддінг тексту через Ollama API."""
    if not text or not text.strip():
        return [0.0] * 4096
    try:
        response = ollama.embeddings(model=MODEL_LLM, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Ollama Embedding Error: {e}")
        return [0.0] * 4096


def calculate_text_metrics(ref, gen, scorer):
    """Рахує BLEU та ROUGE."""
    if not gen or not gen.strip():
        return 0.0, 0.0, 0.0, 0.0

    # Tokenization for BLEU
    ref_tokens = nltk.word_tokenize(ref.lower())
    gen_tokens = nltk.word_tokenize(gen.lower())

    # Smoothing function для уникнення нулів на коротких текстах
    chencherry = SmoothingFunction()
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=chencherry.method1)

    # ROUGE
    rouge_scores = scorer.score(ref, gen)

    return (
        bleu,
        rouge_scores['rouge1'].fmeasure,
        rouge_scores['rouge2'].fmeasure,
        rouge_scores['rougeL'].fmeasure
    )


# ==========================================
# 3. ГОЛОВНИЙ PIPELINE
# ==========================================
def main():
    print("--- STARTING EVALUATION PIPELINE ---")

    # 1. Завантаження компонентів
    try:
        model, vocab, transform = load_inference_components()
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        return

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # 2. Завантаження даних
    print(f"Loading assignments from {FILE_ASSIGNMENTS}...")
    with open(FILE_ASSIGNMENTS, 'r', encoding='utf-8') as f:
        assignments = json.load(f)

    # Нормалізація структури assignments
    normalized_assignments = []
    if isinstance(assignments, list):
        if len(assignments) > 0:
            sample = assignments[0]
            if isinstance(sample, dict):
                # Сподіваємось, що там є ключі 'category' та 'image_path'
                normalized_assignments = assignments
            elif isinstance(sample, list):
                # Формат [path, category]
                normalized_assignments = [{'image_path': x[0], 'category': x[1]} for x in assignments]

    df_assign = pd.DataFrame(normalized_assignments)
    if df_assign.empty or 'category' not in df_assign.columns:
        print("Error: Could not parse assignments DataFrame.")
        return

    print(f"Loading recipes from {FILE_RECIPES}...")
    with open(FILE_RECIPES, 'r', encoding='utf-8') as f:
        raw_recipes = json.load(f)
        # Робимо мапу для швидкого пошуку
        if isinstance(raw_recipes, list):
            # Шукаємо поле з назвою
            sample = raw_recipes[0]
            key = 'recipe_name' if 'recipe_name' in sample else 'title'
            recipes_map = {r[key]: r for r in raw_recipes}
        else:
            recipes_map = raw_recipes

    # 3. Вибір категорій
    top_categories = df_assign['category'].value_counts().head(TOP_N_CATEGORIES).index.tolist()
    print(f"Selected Top {len(top_categories)} Categories: {top_categories}")

    # 4. Підготовка CSV
    headers = [
        'Category', 'Image_ID', 'Strategy',
        'BLEU', 'ROUGE1', 'ROUGE2', 'ROUGEL', 'Cosine_Similarity',
        'Generated_Text', 'Reference_Text', 'Prompt_Text'
    ]

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    print(f"Results will be saved to: {OUTPUT_CSV}")
    print(f"Processing {SAMPLES_PER_CATEGORY} samples per category...")

    # 5. Цикл по категоріях
    for category in tqdm(top_categories, desc="Categories"):
        # Вибираємо випадкові семпли
        cat_samples = df_assign[df_assign['category'] == category]
        n_samples = min(SAMPLES_PER_CATEGORY, len(cat_samples))
        selected_samples = cat_samples.sample(n=n_samples, random_state=42).to_dict('records')

        for sample in selected_samples:
            img_path = sample.get('image_path')

            # Шукаємо Ground Truth рецепт
            recipe_obj = None

            # Спроба 1: Якщо є пряме посилання в assignments
            if 'best_match_recipe_name' in sample and sample['best_match_recipe_name'] in recipes_map:
                recipe_obj = recipes_map[sample['best_match_recipe_name']]

            # Спроба 2: Будь-який рецепт цієї категорії (Fallback)
            if not recipe_obj:
                candidates = [r for r in recipes_map.values() if r.get('category') == category]
                if candidates:
                    recipe_obj = random.choice(candidates)

            if not recipe_obj:
                # Якщо взагалі немає рецепта для цієї категорії, пропускаємо
                continue

            # Формуємо еталонний текст
            ref_text = format_ground_truth(
                recipe_obj.get('recipe_name', category),
                recipe_obj.get('ingredients', []),
                recipe_obj.get('instructions', '')
            )
            # Кешуємо вектор еталону
            emb_ref = get_embedding(ref_text)
            vec_ref = np.array(emb_ref).reshape(1, -1)

            # --- DENSENET INFERENCE ---
            detected_ingredients = predict_ingredients(model, vocab, transform, img_path)

            # Формуємо рядок для промпта: "tomato (0.95), dough (0.88)..."
            det_str = "\n".join([f"* {ing} (Confidence: {conf:.2f})" for ing, conf in detected_ingredients])
            if not det_str:
                det_str = "None detected (Low confidence)"

            # --- PROMPT PREPARATION ---

# Спільні правила форматування (щоб BLEU не скакав)
            format_instruction = """
STRICT OUTPUT FORMAT RULES:
1. STRUCTURE:
[Dish Name]

Ingredients:
1. [Ingredient 1]
2. [Ingredient 2]
...

Instructions:
[Full cooking instructions text]

2. TEXT CLEANING:
- Capitalize the first letter of ingredients and sentences (e.g. "Apple", not "apple").
- REMOVE all confidence scores (e.g. "(0.95)", "(Confidence: ...)") from the output.
- DO NOT output conversational filler (e.g. "Here is the recipe"). Start directly with the Dish Name.
"""

            # ---------------------------------------------------------
            # STRATEGY 1: BASELINE (Professional Standard)
            # ---------------------------------------------------------
            prompt_baseline = f"""System: You are a Michelin-star Executive Chef.
User:
Task: Write a detailed, authentic, and standard recipe for the dish: **{category}**.
Goal: Create the "Ground Truth" version of this dish as it would appear in a professional cookbook. Use standard ingredients associated with this dish name.

{format_instruction}
Assistant:"""

# ---------------------------------------------------------
            # STRATEGY 2: VISUAL CONTEXT (AGGRESSIVE + QUANTITIES FIXED)
            # ---------------------------------------------------------
            prompt_visual = f"""System: You are a Michelin-star Executive Chef.
User:
Target Dish Category: **{category}**
Visual Evidence (Detected Ingredients):
{det_str}

**CRITICAL MISSION:**
The Standard Recipe for {category} is NOT enough. The image shows a **specific variation** of this dish. Your goal is to reconstruct THAT specific variation.

**RULES:**
1. **Detect Variation:** If you see ingredients like 'chocolate', 'walnuts', 'berries' that imply a specific flavor, CHANGE the recipe to match (e.g. "Chocolate {category}").
2. **INFER QUANTITIES (IMPORTANT):** The visual sensor detects presence, NOT amount. You MUST assign standard, plausible quantities to every ingredient.
   - WRONG: "Flour", "Sugar", "Eggs"
   - RIGHT: "2 cups All-purpose Flour", "1/2 cup Granulated Sugar", "2 large Eggs"
3. **Aggressive Integration:** Use the detected ingredients. If 'pecans' are detected, put them in the ingredients list with a quantity (e.g. "1/2 cup chopped pecans").
4. **Clean Output:** Remove confidence scores (0.99), but KEEP cooking numbers (1/2 cup).

{format_instruction}
Assistant:"""

            # --- GENERATION & EVALUATION LOOP ---
            strategies = [
                ('Baseline', prompt_baseline),
                ('VisualContext', prompt_visual)
            ]

            for strat_name, prompt in strategies:
                try:
                    # Call Ollama
                    start_time = time.time()
                    resp = ollama.chat(model=MODEL_LLM, messages=[{'role': 'user', 'content': prompt}])
                    gen_text = resp['message']['content']
                    # print(f"  > Generated {strat_name} in {time.time() - start_time:.1f}s")
                except Exception as e:
                    print(f"  > Ollama Error ({strat_name}): {e}")
                    gen_text = ""

                # Metrics
                bleu, r1, r2, rl = calculate_text_metrics(ref_text, gen_text, scorer)

                # Cosine Similarity
                emb_gen = get_embedding(gen_text)
                if emb_gen:
                    vec_gen = np.array(emb_gen).reshape(1, -1)
                    # Перевірка на нульові вектори
                    if np.any(vec_ref) and np.any(vec_gen):
                        cosine = cosine_similarity(vec_ref, vec_gen)[0][0]
                    else:
                        cosine = 0.0
                else:
                    cosine = 0.0

                # Write to CSV
                with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writerow({
                        'Category': category,
                        'Image_ID': os.path.basename(img_path),
                        'Strategy': strat_name,
                        'BLEU': bleu,
                        'ROUGE1': r1,
                        'ROUGE2': r2,
                        'ROUGEL': rl,
                        'Cosine_Similarity': cosine,
                        'Generated_Text': gen_text,
                        'Reference_Text': ref_text,
                        'Prompt_Text': prompt
                    })

    print("\n" + "=" * 50)
    print(f" PIPELINE FINISHED. RESULTS SAVED TO: {OUTPUT_CSV}")
    print("=" * 50)


if __name__ == '__main__':
    main()