наш таргет 'SalePrice'
В питоне это делается так:
df = pd.read_csv('путь')
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']