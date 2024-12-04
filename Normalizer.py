from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer


class Normalizer:
    def __init__(self):
        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = RobustScaler()
        self.scaler = QuantileTransformer()

    def fit(self, train_data):
        """훈련 데이터로 정규화 기준 학습"""
        self.scaler.fit(train_data)

    def transform(self, data):
        """데이터를 정규화"""
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        """정규화된 데이터를 원본 값으로 복원"""
        return self.scaler.inverse_transform(data)


if __name__ == "__main__":
    # 예제 데이터
    train = np.array([[1, 200], [2, 300], [3, 400], [4, 500], [5, 600]])  # 훈련 데이터
    valid = np.array([[2, 250], [3, 350]])  # 검증 데이터
    test = np.array([[1, 220], [4, 520]])  # 테스트 데이터

    # Normalizer 클래스 인스턴스 생성
    normalizer = Normalizer()

    # 훈련 데이터를 기준으로 정규화 기준 학습
    normalizer.fit(train)

    # 정규화 수행
    train_normalized = normalizer.transform(train)
    valid_normalized = normalizer.transform(valid)
    test_normalized = normalizer.transform(test)

    # 복원
    train_restored = normalizer.inverse_transform(train_normalized)
    valid_restored = normalizer.inverse_transform(valid_normalized)
    test_restored = normalizer.inverse_transform(test_normalized)

    # 출력 결과
    print("원본 데이터 (Train):")
    print(train)

    print("\n정규화된 데이터 (Train):")
    print(train_normalized)

    print("\n복원된 데이터 (Train):")
    print(train_restored)

    print("\n정규화된 데이터 (Valid):")
    print(valid_normalized)

    print("\n복원된 데이터 (Valid):")
    print(valid_restored)

    print("\n정규화된 데이터 (Test):")
    print(test_normalized)

    print("\n복원된 데이터 (Test):")
    print(test_restored)