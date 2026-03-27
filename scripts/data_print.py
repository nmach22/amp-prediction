import pandas as pd

df = pd.read_csv('data/raw/dbaasp_raw.csv')

print('Shape       :', df.shape)
print('Columns     :', list(df.columns))
print('activity val:', df['activity'].value_counts().to_dict())
print('seq len stats:')
print(df['sequence'].str.len().describe().round(1).to_string())
print()
print(df.head(5).to_string())

# print non-AMP rows
print((df['activity'] == 0).sum(), 'non-AMP rows:')
print(df[df['activity'] == 0].to_string())