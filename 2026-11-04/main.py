from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1
def calc_centroid(x):
    n = len(x)
    return (1 / n) * np.sum(x, axis=0)

digits = load_digits()

print(f"Rozmiar zbioru danych: {digits.data.shape[0]} rekordów (obrazów)")

data_train, data_test, target_train, target_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=3)

images_sorted_by_digits = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
}

# pogrupowanie obrazów względem cyfr
for i in range(len(data_train)):
    images_sorted_by_digits[target_train[i]].append(data_train[i])

centroids = {}

# wyznaczenie średnich obrazów dla poszczególnych cyfr
for digit, images in images_sorted_by_digits.items():
    centroids[digit] = calc_centroid(images)

# obrazy centroidów
plt.figure(figsize=(10,5))
plt.suptitle("Obrazy centroidów")

for i in range(10):
    plt.subplot(2, 5,  i+1)
    plt.imshow(centroids[i].reshape(8,8), cmap="gray")
    plt.title(f"Etykieta: {i}")

plt.show()

# 2
def euclides_dist(x, y):
    return np.sqrt(np.sum(np.power((x - y), 2)))

def nearest_centroid(centroids, data):
    min_dist = np.inf
    result_digit = None

    for digit, image in centroids.items():
        dist = euclides_dist(data, image)

        if dist < min_dist:
            min_dist = dist
            result_digit = digit

    return result_digit

results = []

for i in range(len(data_test)):
    results.append(nearest_centroid(centroids, data_test[i]))

# check results
correct_image_count = 0

for i in range(len(data_test)):
    if results[i] == target_test[i]:
        correct_image_count += 1

accuracy = correct_image_count / len(data_test)

print("Skuteczność klasyfikacji:", accuracy)

# 3
def knn(k, data, data_train, target_train):
    # liczenie odległości do sąsiadów
    neighbors = []
    
    for i in range(len(data_train)):
        neighbors.append((euclides_dist(data, data_train[i]), target_train[i]))

    # wybór k najbliższych
    neighbors.sort()
    
    targets = {}

    for i in range(k):
        if targets.get(neighbors[i][1]):
            targets[neighbors[i][1]] += 1
        else:
            targets[neighbors[i][1]] = 1

    # wybór najczęstszego
    most_target = None
    most_target_count = 0

    for key, value in targets.items():
        if most_target_count < value:
            most_target_count = value
            most_target = key

    return most_target


origin_data_train = data_train
origin_target_train = target_train

k_list = [3,20,50]
data_train_sizes = np.arange(300, 1401, 100, dtype=int)

fig, ax = plt.subplots(figsize=(9,7))

# KNN
# print("KNN")
# for k in k_list:
#     print("k =", k)

#     accuracies = []
    
#     for data_train_size in data_train_sizes:
#         data_train = origin_data_train[:data_train_size]
#         target_train = origin_target_train[:data_train_size]

#         # knn
#         results = []

#         for data in data_test:
#             results.append(knn(k, data, data_train, target_train))

#         # check results
#         correct_image_count = 0

#         for i in range(len(data_test)):
#             if results[i] == target_test[i]:
#                 correct_image_count += 1

#         accuracy = correct_image_count / data_test.data.shape[0]
#         accuracies.append(accuracy)


#     ax.plot(data_train_sizes, accuracies, label=f"KNN (k={k})")

# Nearest Centroid
print("Nearest Centroid")

accuracies = []

for data_train_size in data_train_sizes:
    data_train = origin_data_train[:data_train_size]
    target_train = origin_target_train[:data_train_size]

    # calc centroids
    images_sorted_by_digits = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
    }

    # pogrupowanie obrazów względem cyfr
    for i in range(len(data_train)):
        images_sorted_by_digits[target_train[i]].append(data_train[i])

    centroids = {}

    # wyznaczenie średnich obrazów dla poszczególnych cyfr
    for digit, images in images_sorted_by_digits.items():
        centroids[digit] = calc_centroid(images)

    # nearest centroid
    results = []

    for data in data_test:
        results.append(nearest_centroid(centroids, data))

    # check results
    correct_image_count = 0

    for i in range(len(data_test)):
        if results[i] == target_test[i]:
            correct_image_count += 1

    accuracy = correct_image_count / len(data_test)
    accuracies.append(accuracy)

ax.plot(data_train_sizes, accuracies, label="Nearest Centroid")

ax.set_title("Zależność skuteczności klasyfikacji od liczby przykładów treningowych")
ax.set_xlabel("Liczba przykładów treningowych")
ax.set_ylabel("Skuteczność klasyfikacji")
ax.grid()
ax.set_xticks(data_train_sizes)
ax.set_ylim(top=1)

ax.legend()
plt.show()

# 4
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC

data_train = origin_data_train
target_train = origin_target_train

clf = SVC(random_state=0)
clf.fit(data_train, target_train)

# 1. Nearest Centroid
# calc centroids
images_sorted_by_digits = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
}

# pogrupowanie obrazów względem cyfr
for i in range(len(data_train)):
    images_sorted_by_digits[target_train[i]].append(data_train[i])

centroids = {}

# wyznaczenie średnich obrazów dla poszczególnych cyfr
for digit, images in images_sorted_by_digits.items():
    centroids[digit] = calc_centroid(images)

# nearest centroid
results = []

for data in data_test:
    results.append(nearest_centroid(centroids, data))

disp = ConfusionMatrixDisplay.from_predictions(
    target_test,
    results,
    display_labels=clf.classes_,
    cmap=plt.cm.Blues
)

disp.ax_.set_title("Nearest Centroid")
disp.ax_.set_xlabel("Sklasyfikowana cyfra")
disp.ax_.set_ylabel("Rzeczywista cyfra")

# 2. KNN
results = []

for data in data_test:
    results.append(knn(3, data, data_train, target_train))

disp = ConfusionMatrixDisplay.from_predictions(
    target_test,
    results,
    display_labels=clf.classes_,
    cmap=plt.cm.Blues
)

disp.ax_.set_title("KNN (k=3)")
disp.ax_.set_xlabel("Sklasyfikowana cyfra")
disp.ax_.set_ylabel("Rzeczywista cyfra")

plt.show()