from flask import Flask, request
from flask_cors import CORS
import time
import nltk
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ai_models.investment_profile_model.notebooks.tokenization_kobert import KoBertTokenizer

# NLTK 다운로드 (필요한 경우)
nltk.download('punkt')

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 모델 로드
model_path = "./models/classifier_model/distilkobert_investment/kobert_fined_model"
#tokenizer = KoBertTokenizer.from_pretrained("monologg/distilkobert") # 삭제
tokenizer = AutoTokenizer.from_pretrained("./models/classifier_model/distilkobert_investment/kobert_fined_model")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

@app.route("/api/users/profile", methods=['POST'])
def analyze_investment_type():
    """AI 모델을 사용하여 투자 성향 분석"""
    data = request.get_json()
    question = data.get('question', '')
    asset_info = data.get('asset_info', '')
    
    # 입력 전처리 및 예측
    inputs = tokenizer(question + " " + asset_info, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions).item()
    
    # 결과 반환
    return {
        "investmentType": model.config.id2label[predicted_class],
        "confidence": predictions[0][predicted_class].item()
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
