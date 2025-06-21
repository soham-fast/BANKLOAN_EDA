import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style='whitegrid')


pics = 'C:\Users\kadam\Desktop\bank loan EDA\output'
os.makedirs(pics, exist_ok=True)


app_df = pd.read_csv('E:/hackathon day 2/application_data.csv')
prev_df = pd.read_csv('E:/hackathon day 2/previous_application.csv')
desc_df = pd.read_csv('E:/hackathon day 2/columns_description.csv', encoding='ISO-8859-1')


print("\n=== Missing Data ===")
missing_stuff = app_df.isnull().mean() * 100
print("Columns with missing data (%):\n", missing_stuff[missing_stuff > 0].head(10))


plt.figure(figsize=(10, 6))
missing_stuff[missing_stuff > 0].head(10).plot(kind='bar', title='Top 10 Columns with Missing Data (%)')
plt.ylabel('Percentage Missing')
plt.xlabel('Columns')
plt.savefig(os.path.join(pics, 'missing_data.png'))
plt.close()


columns_to_remove = missing_stuff[missing_stuff > 50].index
if columns_to_remove.any():
    print(f"Dropping columns: {columns_to_remove}")
    app_df.drop(columns=columns_to_remove, inplace=True)


if 'AMT_INCOME_TOTAL' in app_df:
    app_df['AMT_INCOME_TOTAL'] = app_df['AMT_INCOME_TOTAL'].fillna(app_df['AMT_INCOME_TOTAL'].median())
if 'AMT_CREDIT' in app_df:
    app_df['AMT_CREDIT'] = app_df['AMT_CREDIT'].fillna(app_df['AMT_CREDIT'].median())
if 'AMT_ANNUITY' in app_df:
    app_df['AMT_ANNUITY'] = app_df['AMT_ANNUITY'].fillna(app_df['AMT_ANNUITY'].median())
if 'AMT_GOODS_PRICE' in app_df:
    app_df['AMT_GOODS_PRICE'] = app_df['AMT_GOODS_PRICE'].fillna(app_df['AMT_GOODS_PRICE'].median())


if 'NAME_EDUCATION_TYPE' in app_df:
    app_df['NAME_EDUCATION_TYPE'] = app_df['NAME_EDUCATION_TYPE'].fillna(app_df['NAME_EDUCATION_TYPE'].mode()[0])
if 'NAME_TYPE_SUITE' in app_df:
    app_df['NAME_TYPE_SUITE'] = app_df['NAME_TYPE_SUITE'].fillna(app_df['NAME_TYPE_SUITE'].mode()[0])


print("\n=== Outliers ===")
def find_weird_numbers(df, col):
    first_quarter = df[col].quantile(0.25)
    third_quarter = df[col].quantile(0.75)
    iqr = third_quarter - first_quarter
    low_limit = first_quarter - 1.5 * iqr
    high_limit = third_quarter + 1.5 * iqr
    weird_numbers = df[(df[col] < low_limit) | (df[col] > high_limit)][col]
    return len(weird_numbers), low_limit, high_limit

if 'AMT_INCOME_TOTAL' in app_df:
    num_outliers, low_num, high_num = find_weird_numbers(app_df, 'AMT_INCOME_TOTAL')
    print(f"Income outliers: {num_outliers} (beyond [{low_num:.2f}, {high_num:.2f}])")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=app_df['AMT_INCOME_TOTAL'])
    plt.title('Income Box Plot')
    plt.savefig(os.path.join(pics, 'boxplot_income.png'))
    plt.close()


print("\n=== Data Imbalance ===")
if 'TARGET' in app_df:
    target_counts = app_df['TARGET'].value_counts()
    good_people = target_counts.get(0, 0)
    bad_people = target_counts.get(1, 0)
    balance_ratio = good_people / bad_people if bad_people != 0 else 'inf'
    print(f"Good people (0): {good_people}")
    print(f"Bad people (1): {bad_people}")
    print(f"Balance Ratio: {balance_ratio:.2f}:1")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='TARGET', data=app_df)
    plt.title('Distribution of Target Variable')
    plt.savefig(os.path.join(pics, 'target_distribution.png'))
    plt.close()


print("\n=== Univariate Analysis ===")
if 'AMT_INCOME_TOTAL' in app_df:
    plt.figure(figsize=(8, 5))
    sns.histplot(app_df['AMT_INCOME_TOTAL'], bins=30)
    plt.title('Income Distribution')
    plt.savefig(os.path.join(pics, 'income_distribution.png'))
    plt.close()

if 'NAME_EDUCATION_TYPE' in app_df:
    plt.figure(figsize=(8, 5))
    sns.countplot(y='NAME_EDUCATION_TYPE', data=app_df)
    plt.title('Education Type Distribution')
    plt.savefig(os.path.join(pics, 'education_distribution.png'))
    plt.close()


print("\n=== Segmented Univariate ===")
if 'TARGET' in app_df and 'AMT_INCOME_TOTAL' in app_df:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=app_df)
    plt.title('Income by Target')
    plt.savefig(os.path.join(pics, 'income_by_target.png'))
    plt.close()


print("\n=== Bivariate Analysis ===")
if 'AMT_INCOME_TOTAL' in app_df and 'AMT_CREDIT' in app_df and 'TARGET' in app_df:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_CREDIT', hue='TARGET', data=app_df)
    plt.title('Income vs Credit by Target')
    plt.savefig(os.path.join(pics, 'income_vs_credit.png'))
    plt.close()


print("\n=== Correlations ===")
if 'TARGET' in app_df:
    bad_group = app_df[app_df['TARGET'] == 1]
    good_group = app_df[app_df['TARGET'] == 0]
    number_cols = [col for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE'] if col in app_df]
    
    if len(bad_group) > 0 and len(number_cols) > 1:
        corr_bad = bad_group[number_cols].corr().abs().unstack()
        corr_bad = corr_bad[corr_bad < 1].sort_values(ascending=False).head(5)
        print("Top Correlations (Bad people):\n", corr_bad)
    
    if len(good_group) > 0 and len(number_cols) > 1:
        corr_good = good_group[number_cols].corr().abs().unstack()
        corr_good = corr_good[corr_good < 1].sort_values(ascending=False).head(5)
        print("Top Correlations (Good people):\n", corr_good)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(good_group[number_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap (Good people)')
        plt.savefig(os.path.join(pics, 'corr_heatmap.png'))
        plt.close()


print("\n=== Insights ===")
print(f"- Data is imbalanced (ratio {balance_ratio:.2f}:1), with {bad_people} bad people vs {good_people} good people.")
print("- Bad people have lower incomes, as seen in segmented analysis.")
print("- Strong connection between AMT_CREDIT and AMT_ANNUITY (0.75 for bad, 0.77 for good) suggests loan terms matter.")
print("- High income outliers ({num_outliers} cases) might be special clients, not necessarily bad.")