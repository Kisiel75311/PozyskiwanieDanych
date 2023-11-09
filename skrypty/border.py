import cv2
import matplotlib.pyplot as plt

# 1. Załadowanie obrazu (bez zmian)
image_path = '2.jpeg'  # aktualizacja ścieżki do obrazu
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
import os

# Tworzenie katalogu tmp jeśli nie istnieje
output_directory = "tmp"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Zakres wartości do przetestowania
block_sizes = list(range(3, 50, 2))  # nieparzyste wartości od 3 do 49
Cs = list(range(-10, 10, 2))  # wartości od -10 do 10 z krokiem 2

for block_size in block_sizes:
    for C in Cs:
        # Zastosowanie adaptacyjnej binaryzacji z daną kombinacją wartości
        adaptive_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY_INV, block_size, C)

        # Zastosowanie operacji morfologicznych
        cleaned_adaptive = cv2.morphologyEx(adaptive_binary, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned_adaptive = cv2.morphologyEx(cleaned_adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Znalezienie konturów
        contours_adaptive, _ = cv2.findContours(cleaned_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Narysowanie konturów na kopii oryginalnego obrazu
        contour_adaptive_image = image.copy()
        cv2.drawContours(contour_adaptive_image, contours_adaptive, -1, (0, 255, 0), 2)

        # Zapis wynikowego obrazu do katalogu tmp
        output_path = os.path.join(output_directory, f"contours_block_{block_size}_C_{C}.png")
        cv2.imwrite(output_path, contour_adaptive_image)

# Wypisanie informacji o zakończeniu procesu
"Zakończono generowanie obrazów dla różnych kombinacji wartości."

