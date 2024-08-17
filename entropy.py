import numpy as np
from collections import Counter

def calculate_shannon_entropy(values):
    """Calculate Shannon Entropy."""
    counts = Counter(values)
    total = len(values)
    if total == 0:
        return 0
    entropy = 0
    for count in counts.values():
        probability = count / total
        entropy -= probability * np.log2(probability)
    return entropy

def calculate_renyi_entropy(values, alpha=2):
    """Calculate Renyi Entropy."""
    if alpha <= 0 or alpha == 1:
        raise ValueError('Alpha must be greater than 0 and not equal to 1')
    
    counts = Counter(values)
    total = len(values)
    if total == 0:
        return 0
    entropy = 0
    for count in counts.values():
        probability = count / total
        entropy += np.power(probability, alpha)
    
    return 1 / (1 - alpha) * np.log2(entropy)

def calculate_column_entropy(df, attributes):
    """Calculate entropy for each specified column in the dataset."""
    column_entropy = {}
    for attr in attributes:
        if attr in df.columns:
            values = df[attr].dropna().tolist()
            if values:
                try:
                    shannon_entropy = calculate_shannon_entropy(values)
                    renyi_entropy = calculate_renyi_entropy(values)
                    column_entropy[attr] = {
                        'shannon': shannon_entropy,
                        'renyi': renyi_entropy
                    }
                except Exception as e:
                    print(f'Error calculating entropy for attribute {attr}: {e}')
                    column_entropy[attr] = {'shannon': None, 'renyi': None}
            else:
                print(f'Warning: No data found for attribute {attr}.')
                column_entropy[attr] = {'shannon': None, 'renyi': None}
        else:
            print(f'Warning: Attribute {attr} not found in DataFrame.')
            column_entropy[attr] = {'shannon': None, 'renyi': None}
    return column_entropy
