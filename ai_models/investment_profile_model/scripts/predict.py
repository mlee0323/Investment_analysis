from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class InvestmentProfilePredictor:
    def __init__(self, model_path="./models/classifier_model/distilkobert-investment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.label_list = ["안정형", "안정추구형", "위험중립형", "공격투자형", "공격형"]
        
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()
        
        return self.label_list[predicted_class]

predictor = InvestmentProfilePredictor()
result = predictor.predict("현재 투자 자금은 전체 금융자산의 40%이며, 총 자산 규모는 5억원 초과, 주식 투자 비중은 70% 초과, 주요 투자 종목은 IT입니다.")
print(result)
