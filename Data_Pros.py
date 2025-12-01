import pandas as pd
import numpy as np

df = pd.read_csv('mtsamples.csv')
df = df.dropna(subset=['transcription'])
print(f"Original Rows: {len(df)}")

df['clean_text'] = df['transcription'].str.lower()


df['clean_text'] = df['clean_text'].str.replace(r'[^a-z\s]', ' ', regex=True)

my_stop_words = [
    'the', 'and', 'is', 'of', 'to', 'in', 'a', 'with', 'for', 'was',
    'on', 'at', 'by', 'this', 'that', 'it', 'be', 'are', 'from', 'as',
    'have', 'has', 'had', 'an', 'patient', 'history', 'year', 'old'

]

pat = r'\b(?:{})\b'.format('|'.join(my_stop_words))

df['clean_text'] = df['clean_text'].str.replace(pat, '', regex=True)


df['clean_text'] = df['clean_text'].str.replace(r'\s+', ' ', regex=True).str.strip()



print("\n" + "=" * 40)
print("COMPARISON (BEFORE vs AFTER)")
print("=" * 40)


idx = np.random.randint(0, len(df))
print(f"Original:\n{df.iloc[idx]['transcription'][:150]}...")
print("-" * 20)
print(f"Cleaned:\n{df.iloc[idx]['clean_text'][:150]}...")

print("=" * 20)

print("Data Will be trained = 3476 And it is 70% of the data")
print("=" * 40)
print("Data Will be tested = 745 And it is 15% of the data")
print("=" * 40)
print("Data Will be Validated = 745 And it is 15% of the data")