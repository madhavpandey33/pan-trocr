import json
import pandas as pd

# Load JSON data
with open('output.log', 'r') as file:
    data = json.load(file)

# Convert data to pandas DataFrame
df = pd.DataFrame(data)

import matplotlib.pyplot as plt

# Plotting the learning rate
plt.figure(figsize=(10, 5))
plt.plot(df['learning_rate'], label='Learning Rate')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Curve')
plt.legend()
plt.show()
