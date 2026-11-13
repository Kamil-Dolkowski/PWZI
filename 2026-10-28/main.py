from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1
digits = load_digits()

print(f"Rozmiar zbioru danych: {digits.data.shape[0]} rekordów (obrazów)")
# print("Przykład obrazu w postaci macieży wartości pikseli: \n", digits.data[0])

plt.figure(figsize=(10,5))
plt.suptitle("Przykładowe obrazy")

for i in range(10):
    plt.subplot(2, 5,  i+1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"Etykieta: {digits.target[i]}")

plt.show()

fig, ax = plt.subplots()
ax.imshow(digits.images[0], cmap="gray")

for i in range(8):
    for j in range(8):
        text = ax.text(j, i, digits.images[0][i, j], ha="center", va="center", color="r")

ax.set_title("Przykład obrazu w postaci macieży wartości pikseli")
fig.tight_layout()

plt.show()

# 2
data_train, data_test, target_train, target_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=3)

# 3
def euclides_dist(x, y):
    return np.sqrt(np.sum(np.power((x - y), 2)))

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

# 4
results = []
k = 3

for data in data_test:
    results.append(knn(k, data, data_train, target_train))

# check results
correct_image_count = 0

for i in range(len(data_test)):
    if results[i] == target_test[i]:
        correct_image_count += 1

accuracy = correct_image_count / data_test.data.shape[0]

print("Skuteczność klasyfikacji: ", accuracy)

# 5
origin_data_train = data_train
origin_target_train = target_train

k_list = [3,20,50]
data_train_sizes = np.arange(300, 1401, 100, dtype=int)

fig, ax = plt.subplots(figsize=(9,7))

for k in k_list:
    print("k =", k)

    accuracies = []
    
    for data_train_size in data_train_sizes:
        data_train = origin_data_train[:data_train_size]
        target_train = origin_target_train[:data_train_size]

        # knn
        results = []

        for data in data_test:
            results.append(knn(k, data, data_train, target_train))

        # check results
        correct_image_count = 0

        for i in range(len(data_test)):
            if results[i] == target_test[i]:
                correct_image_count += 1

        accuracy = correct_image_count / data_test.data.shape[0]
        accuracies.append(accuracy)


    ax.plot(data_train_sizes, accuracies, label=f"k={k}")

ax.set_title("Zależność skuteczności klasyfikacji od liczby przykładów treningowych")
ax.set_xlabel("Liczba przykładów treningowych")
ax.set_ylabel("Skuteczność klasyfikacji")
ax.grid()
ax.set_xticks(data_train_sizes)
ax.set_ylim(top=1)

ax.legend()
plt.show()