import json
import pandas as pd

data = pd.read_csv('../conflict_escalation_dataset.csv')

exclude = ['escalation', 'dispnum', 'statea', 'stateb', 'year']
indicators = []
for col in data.columns:
    if col not in exclude:
        indicators.append(col)

dataset = data[indicators + ['escalation']].copy()

SPLIT_YEAR = 2000 # we will train on older data and test on newer data
train = dataset[data['year'] < SPLIT_YEAR].reset_index(drop=True)

# Calculate statistics for normalization (excluding x0 and escalation)
stats = {}
for col in indicators:
    if col in train.columns:  # Skip if not in train
        stats[col] = {
            'mean': float(train[col].mean()),
            'std': float(train[col].std())
        }

# Save to file
with open('normalization_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("Saved normalization statistics to normalization_stats.json")
