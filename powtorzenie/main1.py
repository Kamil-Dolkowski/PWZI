import matplotlib.pyplot as plt
import numpy as np

dates, smartphones, headsets, tablets, laptops = np.loadtxt(
    "sales.csv", 
    dtype={
        "names": ("date", "smartphones", "headsets", "tablets", "laptops"), 
        "formats": ("datetime64[D]", "i4", "i4", "i4", "i4")
    }, 
    delimiter=",", 
    skiprows=1, 
    unpack=True
)

# 2
def mov_avg(data, window_size):
    return np.array([np.mean(smartphones[i:i+window_size]) for i in range(0, len(smartphones)-window_size)])
smartphones_7avg = mov_avg(smartphones, 7)

def moving_avg(data, window_size):
    results = np.empty(data.size - window_size)

    for i in range(data.size - window_size):
        results[i] = np.average(data[i:i+window_size])

    return results

smartphones_test = moving_avg(smartphones, 7)

print(np.array_equal(smartphones_7avg, smartphones_test))

# 3

all_products = smartphones + headsets + tablets + laptops

all_by_weeks = np.array([np.sum(all_products[i:i+7]) for i in range(0, len(all_products), 7)])
weeks_labels = np.arange(1, all_by_weeks.size+1, 1)

fig, ax = plt.subplots(figsize=(9,7))

ax.bar(weeks_labels, all_by_weeks)

plt.show()

# 4
from matplotlib.patches import Patch

weekdays = (dates.astype(int) + 3)% 7
products_by_weekday = np.array([np.sum(all_products[weekdays == i]) for i in range(0,7)])

bar_colors = ["C0"] * 7
max_id = np.argmax(products_by_weekday)
min_id = np.argmin(products_by_weekday)
bar_colors[max_id] = "green"
bar_colors[min_id] = "red"

fig, ax = plt.subplots(figsize=(9,7))

ax.bar(["pon", "wto", "śro", "czw", "pią", "sob", "nie"], products_by_weekday, color=bar_colors)

ax.legend(handles=[
    Patch(color="green", label="max"),
    Patch(color="red", label="min")
])

plt.show()