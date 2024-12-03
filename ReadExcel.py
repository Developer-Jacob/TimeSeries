import pandas as pd


def read_csv_to_dataframe(file_path):
    """
    CSV 파일을 읽어와 데이터프레임(DataFrame)으로 변환합니다.

    :param file_path: CSV 파일 경로 (str)
    :return: 변환된 데이터프레임 (pd.DataFrame)
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', "Interval", "Vwap", "Direction"]
        return df
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        print(f"CSV 파일을 읽는 도중 오류가 발생했습니다: {e}")


# 사용 예제
if __name__ == "__main__":
    # CSV 파일 경로
    file_path = "XBTUSD_FIVE_MINUTES.csv"  # 읽고자 하는 CSV 파일의 경로 입력
    dataframe = read_csv_to_dataframe(file_path)

    # 결과 출력
    if dataframe is not None:
        print(dataframe)
        dataframe.columns = ['Timestamp', 'Open', 'High', 'Low', 'Volume', 'Close', "Interval", "Vwap", "Direction"]
        print(dataframe.columns)
