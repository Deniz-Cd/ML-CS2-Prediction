"""Plot predicted vs. actual kills from results/predictions.xlsx.

Shows each prediction as a point against the perfect-prediction line
(y = x) with +/- 5 kill margin lines, the tolerance used by the
within-5-kills accuracy metric.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel("results/predictions.xlsx")

# Shared axis limits with a little padding.
all_kills = pd.concat([df["actual_kills"], df["predicted_kills"]])
min_val, max_val = all_kills.min() - 1, all_kills.max() + 1
x_line = np.linspace(min_val, max_val, 100)

plt.figure(figsize=(10, 8))
plt.scatter(df["actual_kills"], df["predicted_kills"], alpha=0.6, s=50, label="Data Points")
plt.plot(x_line, x_line, color="red", linewidth=2, label="Perfect Prediction")
plt.plot(x_line, x_line + 5, color="green", linestyle="--", linewidth=1, label="+5 Margin")
plt.plot(x_line, x_line - 5, color="green", linestyle="--", linewidth=1, label="-5 Margin")

plt.xlabel("Actual Kills")
plt.ylabel("Predicted Kills")
plt.title(r"Model Performance with $\pm 5$ Margin of Error")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
