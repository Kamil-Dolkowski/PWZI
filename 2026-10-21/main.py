import matplotlib.pyplot as plt
import numpy as np
import math

# 1
cars = np.loadtxt("cars.csv", dtype="int", skiprows=1)

figure, axes = plt.subplots(figsize=(10,6))
axes.hist(cars, bins=np.unique(cars).size, range=(np.min(cars)-0.5, np.max(cars)+0.5), edgecolor="black", label="dane pomiarowe")

plt.title("Liczba samochodów przejeżdżających przez skrzyżowanie")
axes.set_xlabel("Liczba samochodów w ciągu minuty")
axes.set_ylabel("Liczba obserwacji")

plt.xticks(np.unique(cars))
plt.grid(visible=True, alpha=0.5)

print(f"Najczęstsze wartości: [na wykresie]")
print(f"Rozrzut wartości: [{np.min(cars)}, {np.max(cars)}]")

# 2
avg_val = np.average(cars)

print("")
print(f"Średnia: {avg_val}")
print(f"Wariancja: {np.var(cars)}")
print(f"Odchylenie standardowe: {np.std(cars)}")

# 3
def poisson(k, _lambda):
    return (np.power(_lambda, k) * np.exp(-_lambda)) / math.factorial(k)

print("\nPrawdopodobieństwa P(X = k):")
for k in np.unique(cars):
    print(poisson(k, avg_val))

# 4
def expected_number_of_observations(k, n):
    return n * poisson(k, avg_val)

n = cars.size
poisson_vals = [expected_number_of_observations(k, n) for k in np.unique(cars)]

axes.stem(np.unique(cars), poisson_vals, markerfmt="C3", linefmt="C3--", label="rozkład Poissona")

plt.legend()
plt.show()

# 5
prob_less_than_15 = sum([poisson(k, avg_val) for k in range(15)])
print(f"\nP(X>=15) = 1 - P(X<=14) = {1 - prob_less_than_15} = {(1 - prob_less_than_15)*100:.2f}%")

# 6
avg_cars_by_hours = np.array([np.average(cars_in_hour) for cars_in_hour in cars.reshape((-1, 60))])

print("\nŚrednia liczba samochodów przypadających na 1 minutę w dla poszczególnych godzin:")
for avg_cars in avg_cars_by_hours:
    print(avg_cars)

figure, axes = plt.subplots(figsize=(9,7))
axes.bar(np.arange(1, avg_cars_by_hours.size + 1, 1), (avg_cars_by_hours - avg_val)/avg_val)

plt.xticks(np.arange(1, avg_cars_by_hours.size + 1, 1))

plt.title("Intensywność ruchu samochodowego w czasie")
axes.set_xlabel("Kolejne godziny")
axes.set_ylabel("Względna zmiana średniej godzinowej względem średniej dla całego zbioru")

plt.grid(visible=True, alpha=0.7)

plt.show()