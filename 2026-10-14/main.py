import numpy as np
import matplotlib.pyplot as plt

WINDOW_SIZE = 7

# Punkty 1, 2

def moving_avg(data, window_size):
    results = np.empty(data.size - window_size)

    for i in range(data.size - window_size):
        results[i] = np.average(data[i:i+window_size])

    return results

dates, smartphones, headsets, tablets, laptops = np.loadtxt(
    "sales.csv", 
    dtype={
        "names": ("date", "smartphones", "headsets", "tablets", "laptops"), 
        "formats": ("S10", "i4", "i4", "i4", "i4")
    }, 
    delimiter=",", 
    skiprows=1, 
    unpack=True
)

figure, axes = plt.subplots(figsize=(9,7))

axes.plot(dates, smartphones, color="C0", linestyle=("dotted"), alpha=0.7)
axes.plot(dates, headsets, color="C1", linestyle=("dotted"), alpha=0.7)
axes.plot(dates, tablets, color="C2", linestyle=("dotted"), alpha=0.7)
axes.plot(dates, laptops, color="C3", linestyle=("dotted"), alpha=0.7)

axes.plot(dates[WINDOW_SIZE:], moving_avg(smartphones, WINDOW_SIZE), label="Smartfony", color="C0")
axes.plot(dates[WINDOW_SIZE:], moving_avg(headsets, WINDOW_SIZE), label="Słuchawki", color="C1")
axes.plot(dates[WINDOW_SIZE:], moving_avg(tablets, WINDOW_SIZE), label="Tablety", color="C2")
axes.plot(dates[WINDOW_SIZE:], moving_avg(laptops, WINDOW_SIZE), label="Laptopy", color="C3")

plt.xticks(np.linspace(0, 365, 12), ["sty", "lut", "mar", "kwi", "maj", "cze", "lip", "sie", "wrz", "paź", "lis", "gru"])

plt.title("Sprzedaż produktów w czasie")
axes.set_xlabel("Dzień roku")
axes.set_ylabel("Ilość sprzedanych produktów [szt]")
axes.legend()

plt.grid(True, alpha=0.6)

plt.show()

# Punkt 3

from matplotlib.patches import Patch

all_products = smartphones + headsets + tablets + laptops
week_count = int(np.ceil(all_products.size/7))
all_products_per_week = np.zeros(week_count)

for i in range(0, 365, 7):
    all_products_per_week[i//7] = np.sum(all_products[i:i+6])

# max, min
max_value_id = np.argmax(all_products_per_week)
min_value_id = np.argmin(all_products_per_week)

figure, axes = plt.subplots(figsize=(9,7))

axes.bar(np.ceil(np.arange(0, 365, 7)/7), all_products_per_week)

plt.title("Łączna sprzedaż w poszczególnych tygodniach")
axes.set_xlabel("Tygodnie")
axes.set_ylabel("Ilość sprzedanych produktów [szt]")

# max, min
bars = axes.patches
bars[max_value_id].set_color("green")
bars[min_value_id].set_color("red")

plt.grid(True, alpha=0.6)

legend_elemants = [
    Patch(color="green", label="Największa sprzedaż"),
    Patch(color="red", label="Najmniejsza sprzedaż"),
]

plt.legend(handles=legend_elemants)

plt.show()

# Punkt 4

dates = dates.astype("datetime64[D]")
dates_by_days = (dates.astype(int) + 3) % 7

products_by_days = np.zeros(7)

for idx, day_id in enumerate(dates_by_days):
    products_by_days[day_id] += all_products[idx]

figure, axes = plt.subplots(figsize=(9,7))
axes.bar(["pon", "wto", "śro", "czw", "pią", "sob", "nie"], products_by_days)

bars = axes.patches
bars[np.argmax(products_by_days)].set_color("green")
bars[np.argmin(products_by_days)].set_color("red")

legend_elemants = [
    Patch(color="green", label="Największa sprzedaż"),
    Patch(color="red", label="Najmniejsza sprzedaż")
]

plt.title("Sprzedaż w poszczególne dni tygodnia")
axes.set_xlabel("Dni tygodnia")
axes.set_ylabel("Ilość sprzedanych produktów [szt]")

plt.grid(True, alpha=0.6)
plt.legend(handles=legend_elemants)

plt.show()