# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# %%

# Pick a dataset and two continuous variables
df = pd.read_csv('ames_prices.csv')


# %%

# Recall the LCLS estimator with the Epanechnikov kernel and the standard plug-in bandwidth for h.
# Epanechnikov formula: 3/4(1 - u^2) where u = (z - x_i) / h
def lcls(x, y, h = None, plot = True):
    n = len(x) # Number of observations
    grid = np.sort(x.unique()) # Extract and sort unique values for x

    # Compute bandwidth, if none provided:
    if h is None:
        iqr = np.quantile(x, .75) - np.quantile(x, .25)
        h = 0.9 * min(np.std(x), iqr/1.34) * len(x) ** (-0.2)
        print(f'Computed bandwidth is: {h}')

    # Compute Epanechnikov kernel
    u = (x.to_numpy().reshape(-1,1)-grid.reshape(1,-1)) / h
    K = np.where(u**2 <= 1, 0.75 * (1 - u**2), 0)

    # Compute LCLS estimator
    numerator = y@K # Compute numerator
    denominator = np.sum(K, axis = 0) # Compute denominator
    y_hat = numerator/denominator # Compute estimator

    # Plot results:
    if plot:
        sns.scatterplot(data = df, y = y, x = x, alpha = .05)
        sns.lineplot(x = grid, y = y_hat, color = 'orange')
        plt.show()

    return y_hat, grid


# %%

# Compute and plot this line for 30 bootstrap samples.
# Notice where there is a lot of variation in the predictions, versus little variation in the predictions.
x = 'area'
y = 'price'

plt.figure(figsize = (10, 6))
sns.scatterplot(x = df[x], y = df[y], alpha = 0.05)

estimates = []
for s in range(30):
    df_s = df.sample(frac = 1.0, replace = True)
    # Compute lcls:
    y_hat_s, grid_s = lcls(df_s[x], df_s[y], plot = False)
    # Plot lcls for each bootstrap sample:
    plt.plot(grid_s, y_hat_s, color = "orange", alpha = 0.1)
    estimates.append((y_hat_s, grid_s))


# Now, for any z, we can bootstrap a distribution of predictions using the above formula.
# Do this at the 25th percentile, median, and 75th percentile of X.






# %%

# Now, pick a grid for z: Obvious choices are all of the unique values in the data, or an equally spaced grid from the minimum value to the maximum value.
# For each z, bootstrap a sample of predictions and compute the .05 and .95 quantiles. Plot these error curves along your LCLS estimate.
# Where are your predictions "tight"/reliable? Where are they highly variable/unreliable?