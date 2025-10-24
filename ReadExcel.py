import pandas as pd


def read_xbtusd_five_to_dataframe():
    """
    CSV 파일을 읽어와 데이터프레임(DataFrame)으로 변환합니다.

    :param file_path: CSV 파일 경로 (str)
    :return: 변환된 데이터프레임 (pd.DataFrame)
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv("data/XBTUSD_FIVE_MINUTES.csv")
        df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', "Interval", "Vwap", "Direction", "Seven_Open", "Seven_High", "Seven_Low", "Seven_Close"]
        return df
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다.: {e}")
    except Exception as e:
        print(f"CSV 파일을 읽는 도중 오류가 발생했습니다: {e}")


# 사용 예제
if __name__ == "__main__":
    # CSV 파일 경로
    dataframe = read_xbtusd_five_to_dataframe()

    # 결과 출력
    if dataframe is not None:
        print(dataframe)
        dataframe.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', "Interval", "Vwap", "Direction", "Seven_Open", "Seven_High", "Seven_Low", "Seven_Close"]
        print(dataframe.columns)
