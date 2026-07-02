import wrds
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv

#calling data from Wharton Research Data Services using WRDS API
#including daily price, volatility, volume data (in PERNMO) for the S&P 500
#rolling 12 month windows for annualized volatility and dollar volume
#returns our features (all z score normalized)

def get_wrds_data():
    load_dotenv()
    db = wrds.Connection(username=os.getenv("WRDS_USERNAME"), password=os.getenv("WRDS_PASSWORD"))

    query = """
        SELECT
            dsf.permno,
            dsf.date,
            ABS(dsf.prc) AS prc,
            dsf.vol,
            dsf.ret
        FROM crsp.dsf AS dsf
    """
    data = db.raw_sql(query, date_cols=["date"])

    db.close()
    return data

def normalize_wrds_data(data):
    frame = data
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["permno", "date"])

    frame["price"] = frame["prc"]
    frame["volume"] = frame["vol"]

    frame["log_return"] = frame.groupby("permno")["price"].transform(
        lambda series: np.log(series).diff()
    )
    frame["volatility"] = frame.groupby("permno")["log_return"].transform(
        lambda series: series.rolling(window=252, min_periods=252).std(ddof=0) * np.sqrt(252)
    )

    for column in ["price", "volume", "volatility"]:
        column_mean = frame[column].mean()
        column_std = frame[column].std(ddof=0)
        if column_std == 0 or np.isnan(column_std):
            frame[f"{column}_zscore"] = 0.0
        else:
            frame[f"{column}_zscore"] = (frame[column] - column_mean) / column_std

    return frame[["permno", "date", "price_zscore", "volume_zscore", "volatility_zscore"]]

