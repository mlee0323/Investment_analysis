import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tokenization_kobert import KoBertTokenizer

# 현재 파일의 절대 경로를 기준으로 모델 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", "classifier_model", "distillkobert_investment", "kobert_fined_model")

# 모델 파일 존재 여부 확인
if not os.path.exists(model_path):
    raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {model_path}")

# KoBertTokenizer 사용
tokenizer = KoBertTokenizer.from_pretrained(
    model_path,
    vocab_file=os.path.join(model_path, "vocab.txt"),
    model_file=os.path.join(model_path, "tokenizer_78b3253a26.model")
)

# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 모델 유형 확인
print(f"Model type: {model.config.model_type}")

# 테스트 데이터 예시
test_data = [
    {
        "responses": [
            {"questionId": 1, "selectedOption": 2},
            {"questionId": 2, "selectedOption": 3},
            {"questionId": 3, "selectedOption": 1},
            {"questionId": 4, "selectedOption": 4},
            {"questionId": 5, "selectedOption": 2},
            {"questionId": 6, "selectedOption": 3},
            {"questionId": 7, "selectedOption": 1}
        ]
    }
]

# 테스트 함수 정의
def test_model(data):
    model.eval()  # 평가 모드로 전환
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for entry in data:
        responses = entry["responses"]

        # 설문 응답 텍스트 생성 (예시 데이터 기반으로 수정 가능)
        input_text = " ".join([f"Q{r['questionId']}:{r['selectedOption']}" for r in responses])

        # 토큰화 및 입력 데이터 생성
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

        # 모델 예측 수행
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_idx = torch.argmax(predictions).item()

        # 결과 출력
        print(f"Investment Type: {model.config.id2label[predicted_class_idx]}")
        print(f"Confidence: {predictions[0][predicted_class_idx].item()}")

# 테스트 실행
test_model(test_data)
