import pandas as pd
import math

def sigmoid(z):
    if z >= 0:
        exp_neg = math.exp(-z)
        return 1 / (1 + exp_neg)
    else:
        exp_pos = math.exp(z)
        return exp_pos / (1 + exp_pos)

data = pd.read_csv('conflict_escalation_dataset.csv')

exclude = ['escalation', 'dispnum', 'statea', 'stateb', 'year']
indicators = []
for col in data.columns:
    if col not in exclude:
        indicators.append(col)

dataset = data[indicators + ['escalation']].copy()

SPLIT_YEAR = 2000 # we will train on older data and test on newer data
train = dataset[data['year'] < SPLIT_YEAR].reset_index(drop=True)
test = dataset[data['year'] >= SPLIT_YEAR].reset_index(drop=True)

train.insert(0, "x0", 1)
test.insert(0, "x0", 1)

print("starting training")

thetas = []
for col in range(train.shape[1] - 1): # exclude 'escalation' column
    thetas.append(0)
for i in range(1000):
    gradients = []
    for col in range(train.shape[1] - 1):
        gradients.append(0)
    for row in train.itertuples(index=False):
        activation = 0
        for k in range(len(thetas)):
            activation += thetas[k] * row[k]
        y = row.escalation
        for m in range(len(gradients)):
            x = row[m]
            gradients[m] += x * (y - sigmoid(activation))
    for n in range(len(thetas)):
        thetas[n] += 0.001 * gradients[n]
    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}/{1000}")

print("\ntraining complete")

param_data = pd.DataFrame({
    "indicator": ["x0"] + indicators,
    "theta": thetas
})
param_data.to_csv("parameters.csv", index=False)

print("\ntesting")

accuracy = 0
for row in test.itertuples(index=False):
    activation = 0
    for n in range(len(thetas)):
        activation += row[n] * thetas[n]
    pr_y = sigmoid(activation)
    if pr_y > 0.5:
        y = 1
    else:
        y = 0
    if y == row.escalation:
        accuracy += 1
print("Accuracy:", accuracy / len(test) * 100, "%")
