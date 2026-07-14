from algorithm.data_caller import get_data
from backtester.get_performance import track_oracle_performance, track_performance

tickers = ["AAPL", "AMZN", "JPM", "JNJ", "XOM", "WMT", "CAT", "NEE", "LIN", "AMT"]


def get_fpso(ticker_list=tickers, epsilon=0.01, iterations=50, timesteps=100):
    data = get_data(tickers=ticker_list)
    return track_performance(data, epsilon=epsilon, iterations=iterations, timesteps=timesteps)


def get_oracle(ticker_list=tickers, timesteps=100, transaction_cost=0.001):
    data = get_data(tickers=ticker_list)
    return track_oracle_performance(data, timesteps=timesteps, transaction_cost=transaction_cost)


if __name__ == "__main__":
    print(get_fpso(tickers))
    print(get_oracle(tickers))