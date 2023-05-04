import pandas as pd

dfs = []
years = list(range(2003, 2022))

for year in years:
    #print(year)
    df = pd.read_csv(f'vectors/{year}_metadata.tsv', sep='\t', header=None, names=['word'])
    dfs.append(df)

df_concat = pd.concat(dfs)

word_counts = df_concat['word'].value_counts()

# convert Series to DataFrame with two columns
df_counts = word_counts.reset_index()
df_counts.columns = ['word', 'count']

count_19 = df_counts[df_counts['count'] == 19]['count'].count()
print('Number of words which appear 19 times: ', count_19)
print(df_counts.head(10))

# 1777 words appear 19 times, 
# Let's look at some of the most frequent AND interesting words from each year, using the year_metadata.tsv file, and see if they appear 19 times
  # If they do, they are candidates for plotting/further analysis

# Words: police, govt, iraq/iraqi, fire, water, court, death, australia, war, attack, drug, election, trade

print(df_counts[df_counts['word'] == 'us']['count'].values[0])

