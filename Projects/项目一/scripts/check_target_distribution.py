import pandas as pd
import numpy as np
from collections import Counter

# Load the dataset
df = pd.read_excel('data/data_aftercalculate.xlsx')

# Check target variable distribution
target_col = 'Secondary Morphology'
print(f"Target variable: {target_col}")
print("=" * 60)

# Count occurrences of each class
class_counts = Counter(df[target_col])
print(f"Total samples: {len(df)}")
print(f"Number of unique classes: {len(class_counts)}")

print("\nClass distribution:")
print("-" * 40)
for class_name, count in sorted(class_counts.items()):
    percentage = (count / len(df)) * 100
    print(f"{class_name}: {count} samples ({percentage:.2f}%)")

# Find classes with only 1 member
single_member_classes = [cls for cls, count in class_counts.items() if count == 1]
print(f"\nClasses with only 1 member: {len(single_member_classes)}")
for cls in single_member_classes:
    print(f"  - {cls}")

# Find classes with less than 2 members (problematic for stratified split)
problematic_classes = [cls for cls, count in class_counts.items() if count < 2]
print(f"\nClasses with less than 2 members: {len(problematic_classes)}")
for cls in problematic_classes:
    print(f"  - {cls}")

# Suggest solutions
print(f"\nSuggested solutions:")
print("1. Remove classes with only 1 member")
print("2. Use random split instead of stratified split")
print("3. Combine rare classes into 'Other' category")
