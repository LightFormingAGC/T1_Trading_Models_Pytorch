import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler


def data_generator(file_path):

    Ticker_csv = pd.read_csv(file_path)
    Ticker_csv.set_index('Date', inplace=True)
    Ticker_csv.index = pd.to_datetime(Ticker_csv.index)
    Ticker_csv['Buy/Sell'] = np.ones(Ticker_csv.shape[0])
    Ticker_csv.drop('Adj Close', axis=1, inplace=True)

    data = Ticker_csv.iloc[80:, :]

    one_day_move = data.pct_change(periods=20).dropna(axis=0)
    long_date_dat = one_day_move[one_day_move['Close'] >= 0.1]
    short_date_dat = one_day_move[one_day_move['Close'] <= -0.1]

    x = []
    y = []
    buy_n = 0
    short_n = 0

    for date in data.index:
        if date in long_date_dat.index:
            post_days = Ticker_csv.loc[:date]
            y.append(np.array([1]))
            buy_n += 1
        elif date in short_date_dat.index:
            post_days = Ticker_csv.loc[:date]
            y.append(np.array([-1]))
            short_n += 1
        else:
            y.append(np.array([0]))
        twenty_days = Ticker_csv.loc[:date].iloc[-80:-20, :].index
        temp_x = []
        for j in range(len(twenty_days)):
            temp_x.append(str(twenty_days[j])[:10])
        x.append(temp_x)
    y = np.array(y)

    x_stats = []
    sc = MinMaxScaler(feature_range=(0, 1))
    for dates in x:
        x_stats.append(sc.fit_transform(Ticker_csv.loc[dates, :'Volume'].values))

    x_stats = np.dstack(x_stats)
    x_stats = np.rollaxis(x_stats, -1)

    # Shuffle Data
    temp = list(zip(x_stats, y))
    random.shuffle(temp)
    x, y = zip(*temp)
    x, y = np.array(x), np.array(y)
    # y = np.expand_dims(y, axis=1)

    return x, y