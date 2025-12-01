import pandas as pd

df = pd.read_pickle('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl')

# Extract the third component from the path (after splitting by '/')
unique_values = df['run_path'].str.split('/').str[3].str.split('-').str[0].unique()

# To see how many unique values you have
num_unique = df['run_path'].str.split('/').str[3].str.split('-').str[0].nunique()

print(f"Number of unique values: {num_unique}")
print(f"Unique values: {unique_values}")
value_counts = df['run_path'].str.split('/').str[3].str.split('-').str[0].value_counts()
print(value_counts)
