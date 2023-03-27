import numpy as np
import pandas as pd
import random
import nolds
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

# zadanie 1
series = np.array(pd.date_range(start="2023-03-01", end="2023-03-30")).reshape(-1, 1)
n = 30
temps = np.zeros(n)
for i in range(n):
    temps[i] = random.randint(-5, 20)

# print(f"Mean: {np.mean(temps)}")
# print(f"Standard deviation: {np.std(temps)}")
# print(f"Max value: {np.max(temps)}")
# print(f"Min value: {np.min(temps)}")
# print(f"Median: {np.median(temps)}")
# print(f"Kurtosis: {kurtosis(temps)}")


# zadanie 2
# series = np.array(pd.date_range(start="2023-03-01", end="2023-03-31")).reshape(-1, 1)
# n = 31
# temps = np.zeros(n)
# for i in range(n):
#     temps[i] = random.randint(-5, 20)

# print(f"Entropy: {nolds.sampen(temps)}")
# print(f"Fractal dimension: {nolds.corr_dim(temps, emb_dim=1)}")
# print(f"Hurst exponent: {nolds.hurst_rs(temps)}")

# zadanie 3
def calc_in_sliding_window(data, func, h):
    beginning_idx = 0
    end_idx = h
    results = []

    while end_idx < len(data):
        results.append(func(data[beginning_idx :  end_idx]))
        beginning_idx += 1
        end_idx += 1

    return results

# print(f"Mean: {calc_in_sliding_window(temps, np.mean, 3)}")

# zadanie 4
def calc_in_sliding_window_standarize_normalize(data, func, h, standarize=False, normalize=False):
    beginning_idx = 0
    end_idx = h
    results = []

    if standarize:
        standarizer = preprocessing.StandardScaler()
        standarized_data = standarizer.fit_transform(np.array(data).reshape(-1, 1))
        while end_idx < len(standarized_data):
            results.append(func(standarized_data[beginning_idx :  end_idx]))
            beginning_idx += 1
            end_idx += 1
        return results
    
    if normalize:
        normalizer = preprocessing.Normalizer(norm='l2')
        normalized_data = normalizer.transform(np.array(temps).reshape(-1, 1))
        while end_idx < len(normalized_data):
            results.append(func(normalized_data[beginning_idx :  end_idx]))
            beginning_idx += 1
            end_idx += 1
        return results
        
    while end_idx < len(data):
        results.append(func(data[beginning_idx :  end_idx]))
        beginning_idx += 1
        end_idx += 1

    return results

fig, ax = plt.subplots(2, 1, sharex=True)
h = 2
standarized_mean = calc_in_sliding_window_standarize_normalize(temps, np.std, h, True, False)
non_standarized_mean = calc_in_sliding_window(temps, np.std, h)
plt.xticks(rotation=30, ha='right')
ax[0].plot(series[:28], np.array(non_standarized_mean).reshape(-1, 1))
ax[0].set_title("Standarized")
ax[1].plot(series[:28], np.array(standarized_mean).reshape(-1, 1))
ax[1].set_title("Non-standarized")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 1, sharex=True)
h = 2
normalized_mean = calc_in_sliding_window_standarize_normalize(temps, np.std, h, False, True)
non_normalized_mean = calc_in_sliding_window(temps, np.std, h)
plt.xticks(rotation=30, ha='right')
ax[0].plot(series[:28], np.array(normalized_mean).reshape(-1, 1))
ax[0].set_title("Normalized")
ax[1].plot(series[:28], np.array(non_normalized_mean).reshape(-1, 1))
ax[1].set_title("Non-normalized")
plt.tight_layout()
plt.show()


