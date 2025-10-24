import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
import numpy as np

columns = ['Open', 'High', 'Low', 'Close', 'Volume', "Interval", "Vwap", "Direction", "Seven_Open", "Seven_High", "Seven_Low", "Seven_Close"]
target_class = "Close"


def get_data_frame(name):
    data = None
    if name == "bit_usd_five":
        data = bit_usd_five()
    elif name == "s_p500":
        data = s_p500()
    else:
        print(f"데이터 이름이 잘못됐음.: {name}")

    print("Data frame: ", name)
    print("Data columns: ", data.columns)
    print("Data columns: ", data.head())
    print("Data frame count: ", len(data))

    return data


def bit_usd_five():
    df = read_excel("XBTUSD_FIVE_MINUTES.csv")
    return df


def s_p500():
    df = fdr.DataReader("S&P500", "1982-04-21")
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan

    # 지정한 순서로 컬럼 재정렬
    df = df[columns]
    return df

def s_p500y():
    data = yf.download('^GSPC', start='1970-01-01', end='2023-12-31').copy()
    data.columns = data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    return data

def read_excel(path):
    try:
        # CSV 파일 읽기
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다.: {path}")
    except Exception as e:
        print(f"CSV 파일을 읽는 도중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    # CSV 파일 경로

    df = s_p500()
    print(df.columns)
    print(df.head())

    df = bit_usd_five()
    print(df.columns)
    print(df.head())