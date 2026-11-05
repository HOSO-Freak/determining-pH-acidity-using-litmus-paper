import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

# --- 1. НАСТРОЙКИ И ЭТАЛОНЫ ---
k = 3  # Количество кластеров для K-means
USE_WHITE_BALANCE = False  # Включи (True), когда будешь фотографировать реальный лакмус на белом фоне

# Названия уровней pH
pH = [
    'pH = 1', 'pH = 2', 'pH = 3', 'pH = 4', 'pH = 5', 'pH = 6', 'pH = 7',
    'pH = 8', 'pH = 9', 'pH = 10', 'pH = 11', 'pH = 12', 'pH = 13', 'pH = 14'
]

# Эталонные цвета шкалы (в формате RGB)
pH_values = [
    (143, 26, 54), (143, 26, 54), (156, 56, 4), (190, 123, 5), (163, 121, 6),
    (165, 142, 5), (102, 93, 6), (60, 75, 14), (20, 48, 18), (29, 25, 51),
    (28, 9, 43), (34, 6, 49), (42, 7, 38), (42, 7, 51)
]

# Создаем словарь для быстрого доступа: {'pH = 1': (143, 26, 54), ...}
reference_dict = dict(zip(pH, pH_values))

# Конвертируем эталонные цвета из RGB в LAB один раз при запуске
reference_lab_dict = {}
for name, rgb in reference_dict.items():
    rgb_array = np.uint8([[rgb]])
    lab_array = cv.cvtColor(rgb_array, cv.COLOR_RGB2LAB)[0][0]
    reference_lab_dict[name] = lab_array

print("Эталоны успешно переведены в LAB.")

# --- 2. ЗАГРУЗКА И ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ ---
print('\nВставьте путь до файла:')
path = input()

image = cv.imread(path)

if image is None:
    print("Ошибка: Не удалось загрузить изображение. Проверьте путь.")
    exit()

# --- 2.1 УМНЫЙ БАЛАНС БЕЛОГО ---
if USE_WHITE_BALANCE:
    print("Применяется баланс белого...")
    # Переводим в LAB для поиска самого светлого участка
    image_lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    L_channel = image_lab[:, :, 0]
    
    # Находим самые яркие пиксели (верхние 5% по яркости)
    max_L = np.max(L_channel)
    threshold_L = max_L - 10 
    white_mask = L_channel > threshold_L
    
    if np.sum(white_mask) > 0:
        # Вычисляем средний оттенок "белого" фона
        avg_A = np.mean(image_lab[:, :, 1][white_mask])
        avg_B = np.mean(image_lab[:, :, 2][white_mask])
        
        # Корректируем всё изображение, чтобы фон стал нейтрально-белым
        image_lab[:, :, 1] = np.clip(image_lab[:, :, 1].astype(np.int16) - int(avg_A - 128), 0, 255).astype(np.uint8)
        image_lab[:, :, 2] = np.clip(image_lab[:, :, 2].astype(np.int16) - int(avg_B - 128), 0, 255).astype(np.uint8)
        
        image = cv.cvtColor(image_lab, cv.COLOR_LAB2BGR)
        print(f"Баланс белого применен. Сдвиг A: {avg_A-128:.1f}, Сдвиг B: {avg_B-128:.1f}")
    else:
        print("Предупреждение: Не найден белый фон для баланса белого.")

# Переводим изображение в оттенки серого для работы порогового алгоритма
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Применяем метод Оцу
ret, mask = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Применяем маску к изображению
result = cv.bitwise_and(image, image, mask=mask)
cv.imwrite("result.jpg", result)

# --- 3. ВЫДЕЛЕНИЕ ПИКСЕЛЕЙ И ПЕРЕВОД В LAB ---
threshold = 5
mask_2 = (result[:,:,0] > threshold) | (result[:,:,1] > threshold) | (result[:,:,2] > threshold)
pixels = result[mask_2].astype(np.float32)

if len(pixels) == 0:
    print("Ошибка: Не найдено ни одного пикселя бумаги. Проверьте качество фото.")
    exit()

# Перевод пикселей в LAB
pixels_reshaped = pixels.reshape(-1, 1, 3).astype(np.uint8)
pixels_lab = cv.cvtColor(pixels_reshaped, cv.COLOR_BGR2LAB).reshape(-1, 3)

# --- 4. КЛАСТЕРИЗАЦИЯ В LAB ---
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(pixels_lab)
centers_lab = kmeans.cluster_centers_ 

unique, counts = np.unique(labels, return_counts=True)

print("\nОсновные цвета (центроиды кластеров в LAB):")
for i, (center, cnt) in enumerate(zip(centers_lab, counts)):
    perc = cnt / len(pixels)
    print(f"Кластер {i+1}: L={center[0]:.1f}, A={center[1]:.1f}, B={center[2]:.1f}, доля={perc:.2f}")

# --- 5. ВЫЧИСЛЕНИЕ СРЕДНЕГО ЦВЕТА ---
weighted_lab = np.average(centers_lab, axis=0, weights=counts)
print(f"\nСредний цвет бумаги (LAB): L={weighted_lab[0]:.1f}, A={weighted_lab[1]:.1f}, B={weighted_lab[2]:.1f}")
# --- 6. СРАВНЕНИЕ С ЭТАЛОНАМИ И ПРОВЕРКА УВЕРЕННОСТИ ---
target_ab = weighted_lab[1:] 
best_name = None
best_dist = float('inf')

for name, ref_lab in reference_lab_dict.items():
    ref_ab = ref_lab[1:] 
    dist = np.linalg.norm(target_ab - ref_ab)
    
    if dist < best_dist:
        best_dist = dist
        best_name = name

# --- 7. ВЫВОД РЕЗУЛЬТАТА С ПРОВЕРКОЙ ---
CONFIDENCE_THRESHOLD = 20.0  # Порог уверенности (если расстояние больше - результат ненадежен)

print("\n--- ИТОГОВЫЙ АНАЛИЗ ---")
if best_dist > CONFIDENCE_THRESHOLD:
    print(f"❌ ВНИМАНИЕ: Цвет не похож ни на один эталон!")
    print(f"   Ближайший вариант: {best_name}, но расстояние слишком велико ({best_dist:.2f} > {CONFIDENCE_THRESHOLD})")
    print("   Возможные причины: плохое освещение, тени, или это не лакмусовая бумага.")
else:
    print(f"✅ РЕЗУЛЬТАТ: {best_name}")
    print(f"   Уверенность: Высокая (расстояние по цвету: {best_dist:.2f})")