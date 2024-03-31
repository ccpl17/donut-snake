from typing import Tuple, Any

import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import plot
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from donut_snake.爬股價 import fetch_stock_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


class DonutSnake:
    # draw_chart:plt

    def __init__(self, ticker, start_date, end_date, look_back, epochs, batch_size, back_month):
        # 傳入參數
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.back_month = back_month  # 回推月份

        #
        self.__train_date_start = None
        self.__train_date_end = None

        self.__train_dataset = None
        self.__sc = None
        self.__X_train_set = None
        self.__X_train = None
        self.__Y_train = None
        self.__model = None
        self.__df_test = None
        self.__X_test_set = None
        self.__X_test = None
        self.Y_test = None
        self.__X_test_set_nor = None
        self.__X_test_nor = None
        self.__Y_test_nor = None
        self.__X_test_nor_reshape = None
        self.__X_test_pred = None
        self.X_test_pred_price = None

        self.mse = None
        self.rmse = None
        self.mae = None
        self.r_squared = None

        self.output = None

        # Run
        self.__指定亂數種子()
        self.__load_train_dataset()
        self.__標準化()
        self.__分割資料()
        self.__轉換測試張量()
        self.__定義模型()
        self.__編譯模型()
        self.__訓練模型()
        self.__載入測試資料()
        self.__產生特徵資料和標籤資料()
        self.__資料標準化()
        self.__標準化後資料()
        self.__轉換訓練張量()
        self.__預測轉回股價()
        self.evaluate()

    # 回推時間設定
    def __回推月份(self):
        # 回推 18 個月
        self.__train_date_end = self.start_date - relativedelta(days=1)

        # 設定起始日期為回推後的日期
        self.__train_date_start = self.start_date - relativedelta(months=self.back_month)

        print(f"回推後的起始日期: {self.__train_date_start} {self.__train_date_end}")
        return self.__train_date_start, self.__train_date_end

    def __指定亂數種子(self) -> None:
        np.random.seed(10)

    # 建立資料集
    def __load_train_dataset(self) -> None:
        self.__train_dataset = fetch_stock_data(self.ticker, self.__回推月份()[0], self.__回推月份()[1])
        self.__X_train_set = self.__train_dataset["Close"].values.reshape(-1, 1)

    # 特徵標準化 - 正規化
    def __標準化(self) -> None:
        self.__sc = StandardScaler()
        self.__X_train_set = self.__sc.fit_transform(self.__X_train_set)

    # 取出幾天前股價來建立成特徵和標籤資料集
    def __create_dataset(self, ds, look_back=1):
        X_data, Y_data = [], []
        for i in range(len(ds) - look_back):
            X_data.append(ds[i:(i + look_back), 0])
            Y_data.append(ds[i + look_back, 0])

        return np.array(X_data), np.array(Y_data)

    # 分割成特徵資料和標籤資料
    def __分割資料(self) -> None:
        self.__X_train, self.__Y_train = self.__create_dataset(self.__X_train_set, self.look_back)

    # 轉換成(樣本數, 時步, 特徵)張量
    def __轉換測試張量(self) -> None:
        print(self.__X_train.size, self.__Y_train.size)
        self.__X_train = np.reshape(self.__X_train, (self.__X_train.shape[0], self.__X_train.shape[1], 1))
        print("X_train.shape: ", self.__X_train.shape)
        print("Y_train.shape: ", self.__Y_train.shape)

    # 定義模型
    def __定義模型(self) -> None:
        self.__model = Sequential()
        self.__model.add(LSTM(50, return_sequences=True,
                              input_shape=(self.__X_train.shape[1], 1)))
        self.__model.add(Dropout(0.2))
        self.__model.add(LSTM(50, return_sequences=True))
        self.__model.add(Dropout(0.2))
        self.__model.add(LSTM(50))
        self.__model.add(Dropout(0.2))
        self.__model.add(Dense(1))
        self.__model.summary()  # 顯示模型摘要資訊

    # 編譯模型
    def __編譯模型(self) -> None:
        self.__model.compile(loss="mse", optimizer="adam")

    # 訓練模型
    def __訓練模型(self) -> None:
        self.__model.fit(self.__X_train, self.__Y_train, epochs=self.epochs, batch_size=self.batch_size)

    # 載入測試資料
    def __載入測試資料(self) -> None:
        self.__df_test = fetch_stock_data(self.ticker, self.start_date, self.end_date)
        self.__X_test_set = self.__df_test["Close"].values.reshape(-1, 1)

        # 複製一份，最終輸出使用
        self.output = self.__df_test

        # 刪垃圾
        self.output.drop(columns=["Open", "High", "Low", "Adj Close", "Volume"], inplace=True)

        # 重新命名 Close 欄，偷吃步法
        self.output["real stock price"] = self.output["Close"]
        self.output.drop(columns=["Close"], inplace=True)
        self.output = self.output.iloc[self.look_back:]

    # 產生特徵資料和標籤資料
    def __產生特徵資料和標籤資料(self) -> None:
        self.__X_test, self.Y_test = self.__create_dataset(self.__X_test_set, self.look_back)
        print(self.Y_test)

    # 資料正規化
    def __資料標準化(self) -> None:
        self.__X_test_set_nor = self.__sc.fit_transform(self.__X_test_set)

    # 產生正規化後的特徵資料和標籤資料
    def __標準化後資料(self) -> None:
        self.__X_test_nor, self.__Y_test_nor = self.__create_dataset(self.__X_test_set_nor, self.look_back)

    # 轉換成(樣本數, 時步, 特徵)張量
    def __轉換訓練張量(self) -> None:
        self.__X_test_nor_reshape = np.reshape(self.__X_test_nor,
                                               (self.__X_test_nor.shape[0], self.__X_test_nor.shape[1], 1))
        self.__X_test_pred = self.__model.predict(self.__X_test_nor_reshape)

    # 將預測值轉換回股價
    def __預測轉回股價(self) -> None:
        self.X_test_pred_price = self.__sc.inverse_transform(self.__X_test_pred)
        # print(type(self.X_test_pred_price))
        # print(self.X_test_pred_price)

        # 將預測結塞回 output
        # self.output["真實股偎"] = self.Y_test
        self.output["Predict stock price"] = self.X_test_pred_price

        # self.output.to_csv("output.csv", index=True)

    def evaluate(self) -> None:
        self.mse = round(mean_squared_error(y_true=self.Y_test, y_pred=self.X_test_pred_price), 4)
        self.rmse = round(mean_squared_error(y_true=self.Y_test, y_pred=self.X_test_pred_price,
                                             squared=False), 4)
        self.mae = round(mean_absolute_error(y_true=self.Y_test, y_pred=self.X_test_pred_price), 4)
        self.r_squared = round(r2_score(y_true=self.Y_test, y_pred=self.X_test_pred_price), 4)


if __name__ == "__main__":
    a = DonutSnake(ticker="8454.TW",
                   # 期間要大於回看天數
                   start_date=datetime.date(2023, 6, 1),
                   end_date=datetime.date(2023, 10, 30),
                   look_back=30,
                   epochs=10,
                   batch_size=32,
                   back_month=18)

    MSE = a.mse
    RMSE = a.rmse
    MAE = a.mae
    R_Squared = a.r_squared

    print(f"MSE：{MSE}")
    print(f"RMSE：{RMSE}")
    print(f"MAE：{MAE}")
    print(f"R-squared：{R_Squared}")

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager

    # 改style要在改font之前
    # plt.style.use("seaborn")

    fontManager.addfont("TaipeiSansTCBeta-Regular.ttf")
    mpl.rc("font", family="Taipei Sans TC Beta")

    plt.figure(figsize=(10, 6))
    plt.plot(a.output['real stock price'], label='real stock price')
    plt.plot(a.output['Predict stock price'], label='Predict stock price')
    plt.title(f"{a.ticker}股價預測")
    plt.xlabel("時間")
    plt.ylabel("股價")
    plt.legend()
    plt.grid(True)
    plt.show()
