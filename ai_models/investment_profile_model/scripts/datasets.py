import pandas as pd
from sklearn.model_selection import train_test_split

def load_new_investment_data(file_path):
    """
    새로운 투자 성향 데이터셋 로드
    """
    data = pd.read_csv(file_path, sep='\t')
    return data

def create_new_investment_dataset():
    """
    새로운 투자 성향 데이터셋 생성 및 분리
    """
    # 1. 데이터 로드
    data = load_new_investment_data("../data/sentiment_data/kobert_investment_data.csv") # 경로 수정

    # 2. 학습, 테스트 데이터 분리
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # 3. train, validation 데이터 분리
    train, val = train_test_split(train, test_size=0.1, random_state=42)

    # 4. CSV 파일로 저장
    train.to_csv("../data/sentiment_data/train.csv", sep='\t', index=False) # 경로 수정
    test.to_csv("../data/sentiment_data/test.csv", sep='\t', index=False) # 경로 수정
    val.to_csv("../data/sentiment_data/validation.csv", sep='\t', index=False) # 경로 수정

    print("데이터셋 생성 완료!")

if __name__ == '__main__':
    create_new_investment_dataset()
