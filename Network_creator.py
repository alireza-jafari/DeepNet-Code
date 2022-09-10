import scipy.sparse as sparse
import pandas as pd
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = pd.read_pickle('data_IRAN_100_updated_1.pkl')


# Build a list of stock collections
column = data.columns
list_of_stocks = pd.DataFrame()
for i in column:
    if i[0] == 'Close':
        dic = {'stocks': i[1]}
        list_of_stocks = list_of_stocks.append(dic, ignore_index=True)

def LDA(x_train, y_train, x_validation, y_validation):
    np.random.seed(1)
    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train)
    score = clf.score(x_validation, y_validation)
    return score
def relu(x):
    if x >= 0:
        return x
    else:
        return 0


START = '2018-01-01'
END = '2021-02-20'

sum_acc = 0
edges = pd.DataFrame(columns=['node-1', 'node-2', 'w'])
for stock_indix in range(len(list_of_stocks['stocks'])):
    stock = list_of_stocks['stocks'].iloc[stock_indix]
    print(stock)
    Stock_data = pd.DataFrame()
    for col in data.columns:
        if col[1] == stock:
            Stock_data[col[0]] = data[col]
    Stock_data = Stock_data[['S_Close', 'S_Volume', 'S_cloes_az_open', 'S_RSI', 'S_BB', 'Y_label',
                             'S_MACD', 'S_SAR', 'S_ADX_DMI', 'S_Stochastic', 'S_MFI', 'S_CCI']]
    Stock_data = Stock_data[Stock_data.index > START]
    Stock_data = Stock_data[Stock_data.index < END]
    Stock_data = Stock_data.dropna()
    lenght_Stock1 = len(Stock_data)

    for stock_neighbour_index in range(len(list_of_stocks['stocks'])):
        stock_neighbour = list_of_stocks['stocks'].iloc[stock_neighbour_index]
        try:
            Stock_data_2 = pd.DataFrame()
            for col2 in data.columns:
                if col2[1] == stock_neighbour:
                    Stock_data_2[col2[0]] = data[col2]
            Stock_data_2 = Stock_data_2[['S_Close', 'S_Volume', 'S_cloes_az_open', 'S_RSI', 'S_BB', 'Y_label',
                                         'S_MACD', 'S_SAR', 'S_ADX_DMI', 'S_Stochastic', 'S_MFI', 'S_CCI']]
            Stock_data_2 = Stock_data_2[Stock_data_2.index > START]
            Stock_data_2 = Stock_data_2[Stock_data_2.index < END]
            Stock_data_2 = Stock_data_2.dropna()

            # -------------------------------------------------------------------------------
            lenght_Stock2 = len(Stock_data_2)
            min = np.minimum(lenght_Stock1, lenght_Stock2)
            max = np.maximum(lenght_Stock1, lenght_Stock2)
            s = max - min

            # -------------------------------------------------------------------------------
            if lenght_Stock2 > lenght_Stock1:
                Stock_data_2 = Stock_data_2[s:]

            X_2 = Stock_data_2.copy()
            X_2 = X_2.drop(['Y_label'], axis=1, inplace=False)

            # 100 days for validation
            train_x_2 = X_2[: len(X_2) - (100)]
            validation_x_2 = X_2[len(X_2) - (100): len(X_2)]

            # -------------------------------------------------------------------------------
            if lenght_Stock1 > lenght_Stock2:
                Stock_data_temp = Stock_data[s:]

                X = Stock_data_temp.copy()
                X = X.drop(['Y_label'], axis=1, inplace=False)
                Y = Stock_data_temp['Y_label']

            else:
                X = Stock_data.copy()
                X = X.drop(['Y_label'], axis=1, inplace=False)
                Y = Stock_data['Y_label']

            # -------------------------------------------------------------------------------

            X_2 = (X + X_2) / 2
            X_2 = X_2.dropna()
            idx = X.index.intersection(X_2.index)
            X = X.loc[idx]
            Y = Y.loc[idx]

            # 100 days for validation
            train_x = X[: len(X) - (100)]
            train_y = Y[: len(Y) - (100)]

            validation_x = X[len(X) - (100): len(Stock_data)]
            validation_y = Y[len(Y) - (100): len(Stock_data)]

            # 100 days for validation
            validation_x_process = X_2[len(X_2) - (100): len(X_2)]
            train_x_process = X_2[: len(X_2) - (100)]

            acc_raw = LDA(train_x, train_y, validation_x, validation_y)
            acc_process = LDA(train_x_process, train_y, validation_x_process, validation_y)
            acc = (acc_process - acc_raw)

            if acc > 0:
                sum_acc += acc
                edge = {'node-1': stock, 'node-2': stock_neighbour, 'w': acc}
                edges = edges.append(edge, ignore_index=True)

        except Exception as e:
            print(e)

print('sum_acc = ', sum_acc)
print(edges)
print(edges['w'].mean())
print(edges['w'].max())
print(edges['w'].sum())

edges['w'] = edges['w'] / edges['w'].max()
edges = edges.sort_values('w', ascending=False)
print(edges['w'])

G = nx.DiGraph()
for i in list_of_stocks['stocks']:
    G.add_node(i)

for index, row in edges.iterrows():
    if row['w'] > 0.1:
        G.add_edge(row['node-1'], row['node-2'], weight=row['w'])



plt.figure(figsize = (10,12))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw(G, pos, node_size = 350, with_labels=True, node_color='lightgreen' )
plt.show()
print(nx.info(G))

temp = nx.adjacency_matrix(G)
A = sparse.csr_matrix(temp)
A = A.astype('float32')
sparse.save_npz('Net_0.1.npz',A)

