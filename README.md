# ML-Life-Expectancy
This program uses Linear Regression to predict average life expectancy based on features such as percentage spend on healthcare and average alcohol consumption. It uses the WHO's dataset on Kaggle.

## Libraries
```bash
pip install pandas scikit-learn
```
The dataset is imported into a dataframe with Pandas.
```bash
# Read the data, display all columns to view all categories for potential features
df = pd.read_csv(r'/Users/maxwellmalinofsky/Desktop/Portfolio/life_expectancy_data.csv')
pd.set_option('display.max.columns', None)
print(df)
```
Some light data cleaning to strip blank spaces found in the category names, ie "Life expectancy  "
```bash
# strip blank spaces from left and right of column names
df.columns = df.columns.str.strip()
print(df.columns
```

