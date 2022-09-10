import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

path = os.listdir(r'C:\Users\Alireza\PycharmProjects\untest\GNN_IRAN\Stocks')
data = pd.DataFrame()
for file in path:
    Stock = pd.read_csv(r'C:\Users\Alireza\PycharmProjects\untest\GNN_IRAN\Stocks'+'/'+ file)
    Stock = Stock.iloc[::-1]
    Stock = Stock.dropna(axis=0)
    Stock['date'] = pd.to_datetime(Stock['<DTYYYYMMDD>'], format='%Y%m%d')
    Stock.set_index('date',inplace=True, drop=True)
    Stock = Stock[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>', '<CLOSE>']]
    Stock.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    for col in Stock.columns:
        data[(col ,file.replace('.csv',''))] = Stock[col]
data = data.dropna(axis=0, how='all')
data = data.dropna(axis=1, how='all')
print(data)

column = data.columns
Stock_list = pd.DataFrame()
for i in column:
    if i[0] == 'Close':
        dic = {'Stock':i[1]}
        Stock_list = Stock_list.append(dic, ignore_index=True)
print(Stock_list)

from sklearn.preprocessing import MinMaxScaler
import talib

def Bollinger_Bands(close, timeperiod, nbdevup, nbdevdn, matype, signals, width):
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn,
                                                    matype=matype)
    df = pd.DataFrame(data=middleband, columns=['middleband'])
    df['upperband'] = upperband
    df['lowerband'] = lowerband
    df['width'] = upperband - lowerband
    df['close'] = close
    df['diff_close'] = np.sign(df['close'].diff(1).shift(0).values)
    df['Bollinger_Bands_signals'] = 0
    for index, row in df.iterrows():
        if row['close'] > (row['upperband'] - (row['width'] * width)) and row['diff_close'] > 0:
            df.at[index, 'Bollinger_Bands_signals'] = +1
        if row['close'] < (row['lowerband'] + (row['width'] * width)) and row['diff_close'] < 0:
            df.at[index, 'Bollinger_Bands_signals'] = -1
    if signals == 1:
        return df['Bollinger_Bands_signals']
    else:
        return df['upperband'], df['middleband'], df['lowerband']
def RSI(close, timeperiod, signals):
    real = talib.RSI(close, timeperiod)
    df = pd.DataFrame(data=real, columns=['RSI'])
    df['RSI_previous'] = df['RSI'].shift(1)
    df['RSI_signals'] = 0
    for index, row in df.iterrows():
        if row['RSI'] <= 30 and row['RSI_previous'] > 30:
            df.at[index, 'RSI_signals'] = +1
        if row['RSI'] >= 70 and row['RSI_previous'] < 70:
            df.at[index, 'RSI_signals'] = -1
    if signals == 1:
        return df['RSI_signals']
    else:
        return df['RSI']
def MACD(close, fastperiod, slowperiod, signalperiod, signals):
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod, slowperiod, signalperiod)
    df = pd.DataFrame(data=macd, columns=['macd'])
    df['macdsignal'] = macdsignal
    df['macdhist'] = macdhist
    df['MACD_signals'] = 0
    for index, row in df.iterrows():
        if row['macd'] > row['macdsignal']:
            df.at[index, 'MACD_signals'] = +1
        if row['macd'] < row['macdsignal']:
            df.at[index, 'MACD_signals'] = -1
    if signals == 1:
        return df['MACD_signals']
    else:
        return df['macd'], df['macdsignal'], df['macdhist']
def SAR(high, low, close, acceleration, maximum, signals):
    real = talib.SAR(high, low, acceleration, maximum)
    df = pd.DataFrame(data=real, columns=['SAR'])
    df['close'] = close
    df['SAR_signals'] = 0
    for index, row in df.iterrows():
        if row['close'] > row['SAR']:
            df.at[index, 'SAR_signals'] = +1
        if row['close'] < row['SAR']:
            df.at[index, 'SAR_signals'] = -1
    if signals == 1:
        return df['SAR_signals']
    else:
        return df['SAR']
def ADX_DMI(high, low, close, timeperiod, signals):
    real = talib.ADX(high, low, close, timeperiod)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod)
    plus_di = talib.PLUS_DI(high, low, close, timeperiod)
    df = pd.DataFrame(data=real, columns=['ADX'])
    df['-DI'] = minus_di
    df['+DI'] = plus_di
    df['diff_ADX'] = np.sign(df['ADX'].diff(1).shift(0).values)
    df['ADX_DMI_signals'] = 0
    for index, row in df.iterrows():
        if row['+DI'] > row['-DI'] and row['diff_ADX'] >= 0 and row['ADX'] > 25:
            df.at[index, 'ADX_DMI_signals'] = +1
        if row['-DI'] > row['+DI'] and row['diff_ADX'] >= 0 and row['ADX'] > 20:
            df.at[index, 'ADX_DMI_signals'] = -1
    if signals == 1:
        return df['ADX_DMI_signals']
    else:
        return df['ADX'], df['+DI'], df['-DI']
def Stochastic(high, low, close, fastk_period, slowk_period, slowd_period, signals):
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period,
                               slowd_period=slowd_period)
    df = pd.DataFrame(data=slowk, columns=['slowk'])
    df['slowd'] = slowd
    df['slowd_previous'] = df['slowd'].shift(1)
    df['slowk_previous'] = df['slowk'].shift(1)
    df['Stochastic_signals'] = 0
    for index, row in df.iterrows():
        if row['slowk'] >= row['slowd'] and row['slowk'] <= 20 and row['slowd'] <= 20 and row['slowk_previous'] < row[
            'slowd_previous']:
            df.at[index, 'Stochastic_signals'] = +1
        if row['slowk'] <= row['slowd'] and row['slowk'] >= 80 and row['slowd'] >= 80 and row['slowk_previous'] > row[
            'slowd_previous']:
            df.at[index, 'Stochastic_signals'] = -1
    if signals == 1:
        return df['Stochastic_signals']
    else:
        return df['slowd'], df['slowk']
def MFI(high, low, close, volume, timeperiod, signals):
    real = talib.MFI(high, low, close, volume, timeperiod=timeperiod)
    df = pd.DataFrame(data=real, columns=['MFI'])
    df['MFI_previous'] = df['MFI'].shift(1)
    df['MFI_signals'] = 0
    for index, row in df.iterrows():
        if row['MFI'] <= 20 and row['MFI_previous'] > 20:
            df.at[index, 'MFI_signals'] = +1
        if row['MFI'] >= 80 and row['MFI_previous'] < 80:
            df.at[index, 'MFI_signals'] = -1
    if signals == 1:
        return df['MFI_signals']
    else:
        return df['MFI']
def CCI(high, low, close, timeperiod, signals):
    real = talib.CCI(high, low, close, timeperiod=timeperiod)
    df = pd.DataFrame(data=real, columns=['CCI'])
    df['CCI_previous'] = df['CCI'].shift(1)
    df['CCI_signals'] = 0
    for index, row in df.iterrows():
        if row['CCI'] >= 100 and row['CCI_previous'] < 100:
            df.at[index, 'MFI_signals'] = +1
        if row['CCI'] <= -100 and row['CCI_previous'] > -100:
            df.at[index, 'CCI_signals'] = -1
    if signals == 1:
        return df['CCI_signals']
    else:
        return df['CCI']


i = 1
for col in Stock_list['Stock']:
    print(i, '/100')
    i += 1

    Stock_data = pd.DataFrame()
    for col2 in data.columns:
        if col2[1] == col:
            Stock_data[(col2[0], col2[1])] = data[col2]
    Stock_data = Stock_data.dropna(axis=0)

    avg_volume = Stock_data.rolling(window=5).mean().shift()[('Volume', col)]
    sign_avg_volume = np.sign(Stock_data[('Volume', col)] - avg_volume)
    data[('S_Volume', col)] = sign_avg_volume

    cloes_yest_close = np.sign(Stock_data[('Adj Close', col)].diff(1))
    data[('S_Close', col)] = cloes_yest_close

    cloes_az_open = np.sign(Stock_data[('Close', col)] - Stock_data[('Open', col)])
    data[('S_cloes_az_open', col)] = cloes_az_open

    data[('S_RSI', col)] = RSI(Stock_data[('Adj Close', col)], 14, 1)
    data[('S_BB', col)] = Bollinger_Bands(Stock_data[('Adj Close', col)], 5, 2, 2, 0, 1, width=0.2)
    data[('S_MACD', col)] = MACD(Stock_data[('Adj Close', col)], 12, 26, 9, 1)
    data[('S_SAR', col)] = SAR(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)], 0.02,
                               0.2, 1)
    data[('S_ADX_DMI', col)] = ADX_DMI(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)],
                                       14, 1)
    data[('S_Stochastic', col)] = Stochastic(Stock_data[('High', col)], Stock_data[('Low', col)],
                                             Stock_data[('Close', col)], 5, 3, 7, 1)
    data[('S_MFI', col)] = MFI(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)],
                               Stock_data[('Volume', col)], 14, 1)
    data[('S_CCI', col)] = CCI(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)], 14, 1)

    # Next Day
    data[('Y_label', col)] = Stock_data[('Close', col)].diff(1).shift(-1)
    data[('Y_label', col)][data[('Y_label', col)] >= 0] = 1
    data[('Y_label', col)][data[('Y_label', col)] < 0] = -1

    data[('Close_scale', col)] = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        data[('Close', col)].values.reshape(len(data), 1))
    data[('High_scale', col)] = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        data[('High', col)].values.reshape(len(data), 1))
    data[('Low_scale', col)] = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        data[('Low', col)].values.reshape(len(data), 1))
    data[('Open_scale', col)] = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        data[('Open', col)].values.reshape(len(data), 1))

    data[('RSI', col)] = talib.RSI(Stock_data[('Adj Close', col)], 14)
    data[('CCI', col)] = talib.CCI(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)],
                                   timeperiod=14)
    data[('MFI', col)] = talib.MFI(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)],
                                   Stock_data[('Volume', col)], timeperiod=14)
    data[('SAR', col)] = talib.SAR(Stock_data[('High', col)], Stock_data[('Low', col)], 0.02, 0.2)
    data[('ADX', col)] = talib.ADX(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)],
                                   timeperiod=14)
    data[('SK', col)], data[('SD', col)] = talib.STOCH(Stock_data[('High', col)], Stock_data[('Low', col)],
                                                       Stock_data[('Close', col)], fastk_period=5, slowk_period=3,
                                                       slowd_period=7)

    data[('MOM', col)] = talib.MOM(Stock_data[('Adj Close', col)], timeperiod=10)
    data[('PPO', col)] = talib.PPO(Stock_data[('Adj Close', col)], fastperiod=12, slowperiod=26, matype=0)
    data[('ROC', col)] = talib.ROC(Stock_data[('Adj Close', col)], timeperiod=10)
    data[('DX', col)] = talib.DX(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)],
                                 timeperiod=14)
    data[('PLUS_DI', col)] = talib.PLUS_DI(Stock_data[('High', col)], Stock_data[('Low', col)],
                                           Stock_data[('Close', col)], timeperiod=14)
    data[('PLUS_DM', col)] = talib.PLUS_DM(Stock_data[('High', col)], Stock_data[('Low', col)], timeperiod=14)
    data[('ADXR', col)] = talib.ADXR(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)],
                                     timeperiod=14)
    data[('WMA', col)] = talib.WMA(Stock_data[('Adj Close', col)], timeperiod=30)
    data[('AD', col)] = talib.AD(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)],
                                 Stock_data[('Volume', col)])
    data[('CO', col)] = talib.ADOSC(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)],
                                    Stock_data[('Volume', col)],
                                    fastperiod=3, slowperiod=10)
    data[('NATR', col)] = talib.NATR(Stock_data[('High', col)], Stock_data[('Low', col)], Stock_data[('Close', col)],
                                     timeperiod=14)
    data[('TRANGE', col)] = talib.TRANGE(Stock_data[('High', col)], Stock_data[('Low', col)],
                                         Stock_data[('Close', col)])
    data[('MACD', col)], data[('MACDsignal', col)], data[('MACDhist ', col)] = talib.MACD(
        Stock_data[('Adj Close', col)], fastperiod=12,
        slowperiod=26, signalperiod=9)

print(data)

data.to_pickle('data_IRAN_100_updated_1.pkl')





