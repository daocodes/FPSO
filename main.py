from algorithm.firefly import *
from algorithm.data_caller import get_wrds_data, normalize_wrds_data
from backtester.get_performance import *

if __name__ == "__main__":
    data=get_wrds_data()
    print(normalize_wrds_data(data))