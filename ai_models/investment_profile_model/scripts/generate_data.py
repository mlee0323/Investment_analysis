import pandas as pd
import random
import os  # os 모듈 추가

# 설문 문항 및 응답 옵션
questions = {
    "Q1": {
        "text": "현재 투자 자금은 전체 금융 자산에서 어느 정도 비중을 차지하나요?",
        "options": ["10% 이하", "10% 이상 ~ 20% 이하", "20% 이상 ~ 30% 이하", "30% 이상 ~ 40% 이하", "40% 초과"]
    },
    "Q2": {
        "text": "주요 수입원은 무엇인가요?",
        "options": ["일정한 수입, 향후 유지 또는 증가 예상", "일정한 수입, 향후 감소 또는 불확실성 예상", "연금 (일정한 수입)", "불규칙한 수입 또는 기타"]
    },
    "Q3": {
        "text": "투자 원금 손실 시 감수할 수 있는 손실 수준은 어느 정도인가요?",
        "options": ["투자 원금 무조건 보전", "10% 미만 손실 감수 가능", "20% 미만 손실 감수 가능", "기대수익이 높다면 위험 감수 가능", "투자 원금의 50% 이상 손실 감수 가능"]
    },
    "Q4": {
        "text": "연령대를 선택해주세요.",
        "options": ["19세 이하", "20세 ~ 40세", "41세 ~ 50세", "51세 ~ 60세", "61세 이상"]
    },
    "Q5": {
        "text": "원하는 투자 기간을 선택해주세요.",
        "options": ["6개월 이내", "6개월 이상 ~ 1년 이내", "1년 이상 ~ 2년 이내", "2년 이상 ~ 3년 이내", "3년 이상"]
    },
    "Q6": {
        "text": "다음 중 경험해본 투자 유형을 모두 선택해주세요.",
        "options": ["예금, 적금", "국공채, 회사채", "주식, 펀드", "ELS"]
    },
    "Q7": {
        "text": "금융 상품에 대한 자신의 지식 수준은 어느 정도인가요?",
        "options": ["투자 경험 없음", "주식과 채권 구분 가능", "대부분 금융 상품 구분 가능", "모든 투자 상품 차이 이해 가능"]
    }
}

# 투자 성향 레이블
investment_styles = ["안정형", "안정추구형", "위험중립형", "공격투자형", "공격형"]

# 응답 옵션별 투자 성향 매핑 (정확도 높이기 위해 수정)
style_mapping = {
    "안정형": {
        "Q1": ["10% 이하"],
        "Q2": ["연금 (일정한 수입)"],
        "Q3": ["투자 원금 무조건 보전"],
        "Q4": ["61세 이상"],
        "Q5": ["6개월 이내"],
        "Q6": ["예금, 적금"],
        "Q7": ["투자 경험 없음"]
    },
    "안정추구형": {
        "Q1": ["10% 이상 ~ 20% 이하", "20% 이상 ~ 30% 이하"],
        "Q2": ["일정한 수입, 향후 유지 또는 증가 예상", "일정한 수입, 향후 감소 또는 불확실성 예상"],
        "Q3": ["10% 미만 손실 감수 가능", "20% 미만 손실 감수 가능"],
        "Q4": ["51세 ~ 60세", "41세 ~ 50세"],
        "Q5": ["6개월 이상 ~ 1년 이내", "1년 이상 ~ 2년 이내"],
        "Q6": ["국공채, 회사채", "예금, 적금"],
        "Q7": ["주식과 채권 구분 가능", "대부분 금융 상품 구분 가능"]
    },
    "위험중립형": {
        "Q1": ["20% 이상 ~ 30% 이하", "30% 이상 ~ 40% 이하"],
        "Q2": ["일정한 수입, 향후 유지 또는 증가 예상"],
        "Q3": ["20% 미만 손실 감수 가능", "기대수익이 높다면 위험 감수 가능"],
        "Q4": ["41세 ~ 50세", "20세 ~ 40세"],
        "Q5": ["1년 이상 ~ 2년 이내", "2년 이상 ~ 3년 이내"],
        "Q6": ["주식, 펀드", "국공채, 회사채"],
        "Q7": ["대부분 금융 상품 구분 가능", "모든 투자 상품 차이 이해 가능"]
    },
    "공격투자형": {
        "Q1": ["30% 이상 ~ 40% 이하", "40% 초과"],
        "Q2": ["일정한 수입, 향후 유지 또는 증가 예상", "불규칙한 수입 또는 기타"],
        "Q3": ["기대수익이 높다면 위험 감수 가능"],
        "Q4": ["20세 ~ 40세", "41세 ~ 50세"],
        "Q5": ["2년 이상 ~ 3년 이내", "3년 이상"],
        "Q6": ["ELS", "주식, 펀드"],
        "Q7": ["모든 투자 상품 차이 이해 가능"]
    },
    "공격형": {
        "Q1": ["40% 초과"],
        "Q2": ["불규칙한 수입 또는 기타"],
        "Q3": ["투자 원금의 50% 이상 손실 감수 가능", "기대수익이 높다면 위험 감수 가능"],
        "Q4": ["20세 ~ 40세"],
        "Q5": ["3년 이상"],
        "Q6": ["ELS", "주식, 펀드"],
        "Q7": ["모든 투자 상품 차이 이해 가능"]
    }
}

# 자산 정보 관련 설정 추가
asset_info_options = {
    "total_assets": ["1천만원 이하", "1천만원 ~ 5천만원", "5천만원 ~ 1억원", "1억원 ~ 5억원", "5억원 초과"],
    "stock_ratio": ["10% 이하", "10% ~ 30%", "30% ~ 50%", "50% ~ 70%", "70% 초과"],
    "main_stock": ["IT", "금융", "제약/바이오", "소비재", "기타"]
}

# 데이터 생성 함수
def generate_investment_data(num_samples=1500):
    data = []
    for style in investment_styles:
        for _ in range(num_samples // len(investment_styles)):
            answers = {}
            for q_id, question in questions.items():
                # 투자 성향에 맞는 응답 선택
                answers[q_id] = random.choice(style_mapping[style][q_id])

            # 자산 정보 추가
            asset_info = {}
            for asset_type, options in asset_info_options.items():
                asset_info[asset_type] = random.choice(options)

            # 텍스트 생성
            text = ""
            for q_id, answer in answers.items():
                text += f"{questions[q_id]['text']}: {answer}. "

            # 자산 정보 텍스트 추가
            text += f"총 자산 규모: {asset_info['total_assets']}. "
            text += f"주식 투자 비중: {asset_info['stock_ratio']}. "
            text += f"주요 투자 종목: {asset_info['main_stock']}. "

            data.append({"text": text, "label": style})

    return pd.DataFrame(data)

# 데이터 생성
investment_data = generate_investment_data(num_samples=7500)

# 결과 확인
print(investment_data.head())

# CSV 파일로 저장
output_path = "../data/sentiment_data/kobert_investment_data.csv" # 경로 수정
investment_data.to_csv(output_path, sep='\t', index=False)

print(f"kobert_investment_data.csv 파일이 {output_path}에 생성되었습니다.")
