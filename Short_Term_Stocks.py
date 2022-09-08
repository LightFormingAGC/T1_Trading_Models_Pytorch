import datetime

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def data_generator(file_path):

    Ticker_csv = pd.read_csv(file_path)
    Ticker_csv.set_index('Date', inplace=True)
    Ticker_csv.index = pd.to_datetime(Ticker_csv.index)
    Ticker_csv['Buy/Sell'] = np.ones(Ticker_csv.shape[0])
    Ticker_csv.drop('Adj Close', axis=1, inplace=True)
    data = Ticker_csv.iloc[21:, :]

    one_day_move = data.pct_change(periods=1).dropna(axis=0)
    long_date_dat = one_day_move[one_day_move['Close'] >= 0.04]
    short_date_dat = one_day_move[one_day_move['Close'] <= -0.04]

    x = []
    y = []
    buy_n = 0
    short_n = 0

    for date in data.index:
        if (date in long_date_dat.index) and (data.loc[date, 'Low'] / data.loc[:date - datetime.timedelta(days=1), 'Close'][-1] >= 0.99):
            post_days = Ticker_csv.loc[:date]
            y.append(np.array([1, 0, 0]))
            buy_n += 1
        elif (date in short_date_dat.index) and (data.loc[date, 'High'] / data.loc[:date - datetime.timedelta(days=1), 'Close'][-1] <= 1.01):
            post_days = Ticker_csv.loc[:date]
            y.append(np.array([0, 0, 1]))
            short_n += 1
        else:
            y.append(np.array([0, 1, 0]))
        twenty_days = Ticker_csv.loc[:date].iloc[-21:-1, :].index
        temp_x = []
        for j in range(len(twenty_days)):
            temp_x.append(str(twenty_days[j])[:10])
        x.append(temp_x)
    y = np.array(y)

    x_stats = []
    sc = StandardScaler()
    for dates in x:
        x_stats.append(sc.fit_transform(Ticker_csv.loc[dates, :'Volume'].values))

    x_stats = np.dstack(x_stats)
    x_stats = np.rollaxis(x_stats, -1)

    y_stats = []
    for i in range(len(y)):
        if i > 19:
            y_stats.append(y[i-20:i])
        else:
            lack = np.array([[0, 1, 0]] * (20 - i))
            y_stats.append(np.concatenate((lack, y[:i])))
    y_stats = np.array(y_stats)

    augmented_long = []
    augmented_long_y = []
    augmented_short = []
    augmented_short_y = []

    for i in range(y_stats.shape[0]):
        if (y_stats[i][-1] == np.array([1, 0, 0])).all():
            original_data = x_stats[i]
            # add noise to every single feature of the data
            noise = np.random.normal(0, 0.05, original_data.shape)
            augmented_long.append(original_data + noise)
            augmented_long_y.append(y_stats[i])
        elif (y_stats[i][-1] == np.array([0, 0, 1])).all():
            original_data = x_stats[i]
            # add noise to every single feature of the data
            noise = np.random.normal(0, 0.05, original_data.shape)
            augmented_short.append(original_data + noise)
            augmented_short_y.append(y_stats[i])

    augmented_short = np.array(augmented_short)
    augmented_short_y = np.array(augmented_short_y)
    augmented_long = np.array(augmented_long)
    augmented_long_y = np.array(augmented_long_y)

    x_stats = np.concatenate((x_stats, augmented_long, augmented_short))
    y_stats = np.concatenate((y_stats, augmented_long_y, augmented_short_y))

    # Shuffle x_stats and y_stats together

    temp = list(zip(x_stats, y_stats))
    random.shuffle(temp)
    x, y = zip(*temp)
    x, y = np.array(x), np.array(y)
    # y = np.expand_dims(y, axis=1)

    # Split Into Train Test Set
    train_n = int(x.shape[0] // (5/4))
    x_train, x_test, y_train, y_test = x[-train_n: ], x[:-train_n], y[-train_n:], y[:-train_n]

    return x_train, y_train, x_test, y_test


def get_long_index(y_test):
    index = []
    for i in range(len(y_test)):
        if (y_test[i][-1] == np.array([1, 0, 0])).all():
            index.append(i)
    return np.array(index)


def get_short_index(y_test):
    index = []
    for i in range(len(y_test)):
        if (y_test[i][-1] == np.array([0, 0, 1])).all():
            index.append(i)
    return np.array(index)


print(get_long_index(data_generator('Data/^IXIC_wk.csv')[3]))
print(get_short_index(data_generator('Data/^IXIC_wk.csv')[3]))
print(data_generator('Data/^IXIC_wk.csv')[3].shape)












