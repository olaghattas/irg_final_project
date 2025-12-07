import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the raw data
raw_data = [
    ["BM25 ($k_1{=}0.9$, $b{=}0.4$)", "0.368 \pm 0.017", "0.436 \pm 0.016"],
    ["LLM Exp + BM25", "0.304 \pm 0.016", "0.38 \pm 0.015"],
    ["LLM Exp + BM25 + LLMRerank", "0.26 \pm 0.0147", "0.324 \pm 0.015"],
    ["LLM Exp +BM25 + DenseRet", "0.348 \pm 0.016", "0.432 \pm 0.015"],
    ["LLM Exp +BM25 + DenseRet + LLMRerank", "0.306 \pm 0.015", "0.38 \pm 0.015"],
    ["TF-IDF (ltc.nnn scratch)", "0.128 \pm 0.003", "0.202 \pm 0.003"],
    ["TF-IDF (sklearn)", "0.225 \pm 0.014", "0.3 \pm 0.014"],
    ["TF-IDF (ltc.nnn scratch) + Cross Encoder", "0.412 \pm 0.018", "0.478 \pm 0.017"],
    ["BM25 + Cross Encoder", "0.418 \pm 0.017", "0.49 \pm 0.016"],
    ["TF-IDF(lnc.nnn)", "0.163 \pm 0.012", "0.228 \pm 0.013"],
    ["Thesaurus Exp + TF-IDF(lnc.nnn)", "0.094 \pm 0.009", "0.148 \pm 0.01"],
    ["Thesaurus Exp + TF-IDF(lnc.nnn) + LLMRerank", "0.171 \pm 0.014", "0.193 \pm 0.014"],
    ["ColBERT", "0.187 \pm 0.014", "0.224 \pm 0.015"],
    ["BM25 + DenseRet", "0.34 \pm 0.016", "0.423 \pm 0.015"],
    ["TF-IDF(nnn.nnn) + LLM Rerank + Krovetz Index", "0.005 \pm 0.002", "0.007 \pm 0.003"],
    ["BM25 + LLM Rerank + Krovetz Index", "0.054 \pm 0.008", "0.064 \pm 0.009"],
    ["TF-IDF(lnc.nnn)+ DenseRet", "0.318 \pm 0.017", "0.355 \pm 0.017"]

]

columns = ["Method", "MAP", "nDCG@50"]
df = pd.DataFrame(raw_data, columns=columns)


df['Method'] = df['Method'].str.replace(r'[\{\}]', '', regex=True).replace({
    '$k_1=0.9$': 'k1=0.9',
    '$b=0.4$': 'b=0.4'
}, regex=True).str.replace(r'[\\$]', '', regex=True)

# 2. Function to parse the "Value \pm Error" strings
def parse_value_error(s):
    try:
        # Split by '\pm' and strip whitespace, then convert to float
        parts = s.split(r'\pm')
        value = float(parts[0].strip())
        error = float(parts[1].strip())
        return value, error
    except:
        return np.nan, np.nan

# 3. Apply the function to MAP and nDCG@50 columns
df[['MAP_Value', 'MAP_Error']] = df['MAP'].apply(lambda x: pd.Series(parse_value_error(x)))
df[['nDCG@50_Value', 'nDCG@50_Error']] = df['nDCG@50'].apply(lambda x: pd.Series(parse_value_error(x)))

# Drop the original columns
df_clean = df.drop(columns=['MAP', 'nDCG@50'])

# 4. Save the resulting DataFrame to a CSV file
csv_filename = "retrieval_performance_data.csv"
df_clean.to_csv(csv_filename, index=False)

# 5. Prepare data for plotting (long format)
# Melt the DataFrame to have one row per metric (MAP or nDCG@50) for easier plotting
df_plot = pd.melt(df_clean, id_vars=['Method'],
                  value_vars=['MAP_Value', 'nDCG@50_Value'],
                  var_name='Metric',
                  value_name='Value')

# Map the error values to the long format
error_map = {
    'MAP_Value': df_clean['MAP_Error'],
    'nDCG@50_Value': df_clean['nDCG@50_Error']
}

errors = []
for index, row in df_plot.iterrows():
    # Find the corresponding error from the original df_clean based on the row's original index and metric
    original_index = index % len(df_clean)
    metric_key = row['Metric'].replace('_Value', '_Error')
    errors.append(df_clean.loc[original_index, metric_key])

df_plot['Error'] = errors

# Rename metrics for cleaner labels
df_plot['Metric'] = df_plot['Metric'].str.replace('_Value', '').str.replace('nDCG@50', 'nDCG@50')

# Sort data by MAP_Value descending for better visualization
map_values = df_clean.set_index('Method')['MAP_Value'].sort_values(ascending=False).index
df_plot['Method'] = pd.Categorical(df_plot['Method'], categories=map_values, ordered=True)
df_plot = df_plot.sort_values(['Method', 'Metric'])


# 6. Create the bar plot with error bars
fig, ax = plt.subplots(figsize=(12, 8))


bar_width = 0.35
methods = df_plot['Method'].cat.categories
ind = np.arange(len(methods))
metrics = df_plot['Metric'].unique()

# Separate data for MAP and nDCG@50
map_data = df_plot[df_plot['Metric'] == 'MAP']
ndcg_data = df_plot[df_plot['Metric'] == 'nDCG@50']

# Plot bars
rects1 = ax.bar(ind - bar_width/2, map_data['Value'], bar_width,
                yerr=map_data['Error'], capsize=5, label='MAP', color='#1f77b4') # Blue
rects2 = ax.bar(ind + bar_width/2, ndcg_data['Value'], bar_width,
                yerr=ndcg_data['Error'], capsize=5, label='nDCG@50', color='#ff7f0e') # Orange

# Add labels, title and custom x-axis tick labels
ax.set_ylabel('Performance Metric Value')
ax.set_title('Retrieval Performance by Method (MAP vs. nDCG@50)')
ax.set_xticks(ind)
ax.set_xticklabels(methods, rotation=90, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_ylim(0, 0.55) # Set a reasonable limit for the y-axis

plt.tight_layout()
plt.savefig('retrieval_performance_bar_graph.png')
