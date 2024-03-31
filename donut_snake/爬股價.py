import yfinance as yf


def fetch_stock_data(ticker, start_date, end_date):
    # 使用 download 函數直接下載歷史數據
    historical_data = yf.download(ticker, start=start_date, end=end_date)

    return historical_data


if __name__ == "__main__":
    ticker = "AAPL"  # 替換為您要爬取的台灣股票代碼
    start_date = "2023-01-01"  # 替換為開始日期
    end_date = "2023-10-30"  # 替換為結束日期

    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # 將歷史股票數據保存到 CSV 檔案
    stock_data.to_csv("stock_data.csv", index=False)

    # 打印歷史股票數據
    print(stock_data)
