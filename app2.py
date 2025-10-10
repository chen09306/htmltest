from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import stanza
from flask_cors import CORS  # 處理跨域

app = Flask(__name__)
CORS(app)  # 允許前端跨域請求

# ====================== 金融問答模型 ======================
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"  

print("正在載入金融模型，請稍候...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": "cpu"},  # 強制 CPU
    torch_dtype=torch.float32,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def financial_qa(question):
    """金融問答系統"""
    system_prompt = (
        "你是一位專業的金融顧問，請用繁體中文以清楚的條列式回覆，"
        "包含解釋與具體建議。"
    )
    prompt = f"{system_prompt}\n\n用戶問題：{question}\n\n請回答："
    result = generator(prompt, max_new_tokens=500, do_sample=True, temperature=0.7, top_p=0.9)
    return result[0]['generated_text'].replace(prompt, "").strip()

@app.route("/ask", methods=["POST"])
def ask():
    """API: 金融問答"""
    question = request.json.get("question", "")
    answer = financial_qa(question)
    return jsonify({"answer": answer})

# ====================== 新聞摘要 ======================
# Stanza 初始化（只要第一次下載模型即可）
stanza.download('zh')  
nlp = stanza.Pipeline('zh')

@app.route("/news", methods=["POST"])
def news_summary():
    """API: 新聞摘要"""
    news_content = request.json.get("question", "")
    sentences = [s for s in news_content.split("。") if s]

    results = []
    for sent_no, sentence in enumerate(sentences):
        doc = nlp(sentence)
        tokens = [word.text for sent in doc.sentences for word in sent.words]
        no_of_tokens = len(tokens)
        no_of_nouns = len([word for sent in doc.sentences for word in sent.words if word.upos in ["NOUN", "PROPN"]])
        no_of_ners = len(doc.entities)
        score = (2*no_of_ners + no_of_nouns) / float(no_of_tokens) if no_of_tokens > 0 else 0
        results.append((sent_no, score, sentence))

    # 取前 3 句作為摘要
    results_3 = sorted(results, key=lambda x: x[1], reverse=True)[:3]
    results_sorted = sorted(results_3, key=lambda x: x[0])
    summary = "\n".join([f"{i+1}. {s[2]}" for i, s in enumerate(results_sorted)])
    return jsonify({"answer": summary})

# ====================== 資產配置 ======================
def suggest_risk_by_age(age=None):
    """
    根據年齡計算股票比例 (110 - 年齡)，其餘配置給債券
    """
    if age is None:
        age = 30  # 預設值
    high = max(0, min(100, 110 - age)) / 100
    low = 1 - high

    if age < 30:
        desc = "現金流穩定，可採取積極型策略"
        tools = "綠電、ETF、定/活存、股票"
    elif 30 <= age <= 45:
        desc = "穩定累積資產，可採取穩健型策略"
        tools = "綠電、指數股票型基金(ETF)、股票、債券、保險、不動產投資信託(REITs)、房地產"
    else:
        desc = "規劃退休，可採取保守型策略"
        tools = "綠電、指數股票型基金(ETF)、股票、房地產"

    return {"高風險": high, "低風險": low}, desc, tools

def get_allocation(age=None, risk_level=None, amount=None):
    """
    資產配置邏輯 (優先使用年齡，其次才是風險偏好)
    """
    if age is not None:
        allocation, desc, tools = suggest_risk_by_age(age)
        result_parts = []
        for k, v in allocation.items():
            money = int(amount * v)
            percent = int(v * 100)
            result_parts.append(f"{k} {money} 元 ({percent}%)")
        result_str = " + ".join(result_parts)

        return (f"依據年齡 {age} 歲\n投資金額 {int(amount)} 元\n"
                f"策略建議：{desc}\n"
                f"建議資產工具：{tools}\n"
                f"配置：{result_str}")

    allocations = {
        "高": {"高風險": 0.8, "低風險": 0.2},
        "中": {"高風險": 0.6, "低風險": 0.4},
        "低": {"高風險": 0.3, "低風險": 0.7}
    }

    if not risk_level or risk_level not in allocations:
        return "請輸入年齡或風險偏好（高/中/低）"

    allocation = allocations[risk_level]
    result_parts = []
    for k, v in allocation.items():
        money = int(amount * v)
        percent = int(v * 100)
        result_parts.append(f"{k} {money} 元 ({percent}%)")
    result_str = " + ".join(result_parts)

    return f"依據風險偏好 {risk_level}\n投資金額 {int(amount)} 元\n配置：{result_str}"

@app.route("/asset", methods=["POST"])
def asset_allocation():
    """API: 資產配置"""
    data = request.json
    age = data.get("age")
    risk_level = data.get("risk_level")
    amount = data.get("amount")

    if not amount:
        return jsonify({"answer": "投資金額必須輸入！"})

    if age is not None:
        try:
            age = int(age)
        except:
            return jsonify({"answer": "年齡格式錯誤，請輸入整數"})

    result = get_allocation(age=age, risk_level=risk_level, amount=float(amount))
    return jsonify({"answer": result})

# ====================== 主程式 ======================
if __name__ == "__main__":
    # 只有在本機開發時才會用到
    app.run(host="0.0.0.0", port=5000, debug=True)


