import matplotlib.pyplot as plt
import numpy as np

# 1
data = np.loadtxt(fname="cars.csv", dtype=(int), skiprows=1)

fig, ax = plt.subplots(figsize=(9,7))

ax.hist(data, np.unique(data).size, align="mid", range=(np.min(data)-0.5, np.max(data)+0.5), edgecolor="black")

ax.set_xticks(np.unique(data))

plt.show()

# 2
avg_v = np.mean(data)
var_v = np.var(data)
std_v = np.std(data)
print(avg_v, var_v, std_v)

# 3
import math 

def poisson(k, lambda_):
    return (np.power(lambda_, k) * np.exp(-lambda_)) / math.factorial(k)

poisson_probabilities = np.array([poisson(k, avg_v) for k in np.unique(data)])

# 4
n = len(data)
poisson_values = n * poisson_probabilities

fig, ax = plt.subplots(figsize=(9,7))

ax.hist(data, np.unique(data).size, align="mid", range=(np.min(data)-0.5, np.max(data)+0.5), edgecolor="black")
ax.scatter(np.unique(data), poisson_values, color="red")

ax.set_xticks(np.unique(data))

plt.show()

# 5
# P(X>=15) = 1 - P(X<=14)
poisson_less_and_equal_14 = np.sum([poisson(k, avg_v) for k in range(15)])
print(1 - poisson_less_and_equal_14)

# 6
avg_data_by_hours = np.array([np.mean(data[i:i+60]) for i in range(0, len(data), 60)])
wzgl_avg =(avg_data_by_hours - avg_v) / avg_v

fig, ax = plt.subplots(figsize=(9,7))

# ax.bar(np.arange(0, avg_data_by_hours.size, 1), avg_data_by_hours)
ax.bar(np.arange(0, avg_data_by_hours.size, 1), wzgl_avg)
ax.set_xticks(np.arange(0, avg_data_by_hours.size, 1), np.arange(0, avg_data_by_hours.size, 1) + 1)

plt.show()