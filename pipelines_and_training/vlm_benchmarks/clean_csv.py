import pandas as pd
csv_path = '/home/haderax/DoAn/official_benchmark/benchmark_results.csv'
df = pd.read_csv(csv_path)
df = df[~df['Model'].isin(['Qwen3-VL (8B) - LoRA v2 (Prompt v2)', 'Qwen3-VL (8B) - Base (Prompt v2)', 'Qwen3-VL (8B) - LoRA v1 (Prompt v1)'])]
df.to_csv(csv_path, index=False)
print('Cleaned test rows. Total rows remaining:', len(df))
print(df['Model'].value_counts())
