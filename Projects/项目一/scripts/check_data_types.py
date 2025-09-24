import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_excel('data/data_aftercalculate.xlsx')

categorical_columns = ['Co-material', 'F-base', 'code', 'Secondary Morphology']

print("Checking for mixed data types in categorical columns:")
print("=" * 60)

for col in categorical_columns:
    print(f"\nColumn: {col}")
    print(f"Data type: {df[col].dtype}")
    
    # Get unique values
    unique_vals = df[col].dropna().unique()
    print(f"Number of unique values: {len(unique_vals)}")
    
    # Check for mixed types
    type_counts = {}
    for val in unique_vals:
        val_type = type(val).__name__
        type_counts[val_type] = type_counts.get(val_type, 0) + 1
    
    print(f"Type distribution: {type_counts}")
    
    # Show some examples of each type
    for val_type in type_counts:
        examples = [val for val in unique_vals if type(val).__name__ == val_type][:3]
        print(f"  {val_type} examples: {examples}")
    
    # Check if there are mixed types
    if len(type_counts) > 1:
        print(f"  ⚠️  MIXED TYPES DETECTED in column '{col}'!")
        print(f"  This will cause LabelEncoder to fail.")
    else:
        print(f"  ✓ Uniform data type: {list(type_counts.keys())[0]}")
