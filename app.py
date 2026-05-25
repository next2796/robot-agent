from flask import Flask, request, jsonify, session
from flask_cors import CORS
import sqlite3
import requests
import time
import json
import hashlib
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app, supports_credentials=True)

# --------------------------
# 1. 数据库初始化（医疗知识库）
# --------------------------
def init_db():
    conn = sqlite3.connect('medical.db')
    c = conn.cursor()
    # 创建疾病知识库表
    c.execute('''
        CREATE TABLE IF NOT EXISTS diseases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            symptoms TEXT,
            suggestion TEXT,
            risk_level TEXT
        )
    ''')
    # 创建健康建议表
    c.execute('''
        CREATE TABLE IF NOT EXISTS health_tips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            title TEXT,
            content TEXT
        )
    ''')
    # 创建对话历史表
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT,
            bot_reply TEXT,
            intent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # 创建用户表
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # 插入示例疾病数据
    disease_data = [
        ("感冒", "发热、咳嗽、流鼻涕、乏力", "多喝水，注意休息，体温超过38.5℃建议就医", "低"),
        ("急性肠胃炎", "腹痛、腹泻、呕吐、恶心", "清淡饮食，补充电解质，严重脱水请立即就医", "中"),
        ("高血压急症", "剧烈头痛、呕吐、视物模糊、血压骤升", "立即卧床休息，拨打120，避免活动", "高"),
        ("糖尿病", "多饮、多食、多尿、体重减轻", "控制饮食，规律运动，定期监测血糖", "中"),
        ("肺炎", "高热、咳嗽、咳痰、呼吸困难", "及时就医，遵医嘱使用抗生素", "高")
    ]
    c.executemany('INSERT OR IGNORE INTO diseases (name, symptoms, suggestion, risk_level) VALUES (?, ?, ?, ?)', disease_data)
    
    # 插入健康建议数据
    tip_data = [
        ("日常保健", "充足睡眠", "建议每天保证7-8小时的睡眠时间，有助于身体恢复和免疫力提升。"),
        ("日常保健", "均衡饮食", "多吃蔬菜水果，减少油腻和高糖食物的摄入。"),
        ("日常保健", "适度运动", "每周进行至少150分钟的中等强度有氧运动。"),
        ("用药安全", "遵医嘱用药", "严格按照医生的建议服用药物，不要自行增减剂量。"),
        ("用药安全", "药物存放", "将药物放在儿童接触不到的地方，定期检查有效期。")
    ]
    c.executemany('INSERT OR IGNORE INTO health_tips (category, title, content) VALUES (?, ?, ?)', tip_data)
    
    conn.commit()
    conn.close()

init_db()

# --------------------------
# 2. NLP引擎：意图识别 + 槽位填充（调用Ollama）
# --------------------------
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"  # Qwen3-8B 模型
TIMEOUT_SECONDS = 60  # 增加超时时间

def check_ollama_status():
    """检查Ollama服务是否可用"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def call_ollama(prompt, max_retries=2):
    """调用本地Ollama模型，支持重试机制"""
    attempts = 0
    while attempts < max_retries:
        try:
            response = requests.post(OLLAMA_API, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 512
                }
            }, timeout=TIMEOUT_SECONDS)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            elif response.status_code == 500:
                # 模型加载中，等待后重试
                time.sleep(5)
                attempts += 1
                continue
            else:
                print(f"Ollama API返回错误状态码: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"模型调用超时，第{attempts+1}次尝试")
            attempts += 1
            time.sleep(2)
        except requests.exceptions.ConnectionError:
            print("Ollama服务连接失败，请检查服务是否启动")
            return None
        except Exception as e:
            print(f"模型调用失败: {e}")
            return None
    
    print(f"模型调用失败，已重试{max_retries}次")
    return None

def intent_recognition(user_input):
    """意图识别：判断用户意图是咨询症状、查询疾病、紧急求助或工具调用"""
    prompt = f"""
    你是一个医疗对话系统的意图识别器，用户输入: {user_input}
    请只返回一个意图类型，可选值：
    - symptom_consult（用户描述症状，咨询可能疾病）
    - disease_query（用户询问已知疾病的信息）
    - emergency（用户描述紧急症状，需风险评估）
    - tool_bmi（用户想计算BMI指数）
    - tool_health_tip（用户想获取健康建议）
    - unknown（无法识别意图）
    仅返回关键词，不要额外解释
    """
    intent = call_ollama(prompt)
    return intent.strip() if intent else "unknown"

def slot_filling(user_input, intent):
    """槽位填充：提取关键信息，如症状、疾病名称、用户情况"""
    if intent == "symptom_consult":
        prompt = f"""
        从用户输入中提取所有症状，用逗号分隔，用户输入: {user_input}
        仅返回症状列表，不要其他内容
        """
        symptoms = call_ollama(prompt)
        return {"symptoms": symptoms.strip() if symptoms else ""}
    elif intent == "disease_query":
        prompt = f"""
        从用户输入中提取疾病名称，用户输入: {user_input}
        仅返回疾病名称，不要其他内容
        """
        disease = call_ollama(prompt)
        return {"disease": disease.strip() if disease else ""}
    elif intent == "tool_bmi":
        prompt = f"""
        从用户输入中提取身高(cm)和体重(kg)，格式: 身高 体重
        用户输入: {user_input}
        仅返回两个数字，用空格分隔
        """
        result = call_ollama(prompt)
        try:
            parts = result.strip().split()
            return {"height": float(parts[0]), "weight": float(parts[1])}
        except:
            return {}
    return {}

# --------------------------
# 3. Agent工具模块
# --------------------------

class BMICalculator:
    """BMI计算器工具 - 计算用户的身体质量指数并给出健康建议"""
    
    @staticmethod
    def calculate(height_cm, weight_kg):
        try:
            height_m = height_cm / 100
            bmi = weight_kg / (height_m ** 2)
            bmi = round(bmi, 1)
            
            if bmi < 18.5:
                category = "偏瘦"
                suggestion = "您的体重偏轻，建议适当增加营养摄入，均衡饮食，适度进行力量训练。"
            elif 18.5 <= bmi < 24:
                category = "正常"
                suggestion = "您的体重在正常范围内，请继续保持健康的生活方式！"
            elif 24 <= bmi < 28:
                category = "超重"
                suggestion = "您的体重略微超标，建议控制饮食，增加运动量，保持健康体重。"
            else:
                category = "肥胖"
                suggestion = "您的体重超标较多，建议咨询医生制定合理的减重计划，注意饮食和运动。"
            
            return {
                "success": True,
                "bmi": bmi,
                "category": category,
                "suggestion": suggestion,
                "range_info": "BMI范围：偏瘦<18.5，正常18.5-24，超重24-28，肥胖≥28"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def execute(user_input):
        result = call_ollama(f"""
        从用户输入中提取身高(cm)和体重(kg)，格式: 身高 体重
        用户输入: {user_input}
        仅返回两个数字，用空格分隔
        """)
        try:
            parts = result.strip().split()
            height = float(parts[0])
            weight = float(parts[1])
            return BMICalculator.calculate(height, weight)
        except:
            return {"success": False, "error": "无法识别身高体重信息，请输入格式如：我身高175厘米，体重65公斤"}

class HealthTipGenerator:
    """健康建议生成器工具 - 根据用户需求提供个性化健康建议"""
    
    @staticmethod
    def get_tips(category=None):
        conn = sqlite3.connect('medical.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        if category:
            c.execute('SELECT * FROM health_tips WHERE category LIKE ?', (f'%{category}%',))
        else:
            c.execute('SELECT * FROM health_tips ORDER BY RANDOM() LIMIT 3')
        
        result = [dict(row) for row in c.fetchall()]
        conn.close()
        return result

    @staticmethod
    def generate_personalized_tip(user_input):
        category = call_ollama(f"""
        分析用户的健康需求，判断用户可能感兴趣的健康主题，如：日常保健、饮食营养、运动健身、睡眠质量、压力管理等
        用户输入: {user_input}
        仅返回一个主题关键词
        """)
        category = category.strip() if category else "日常保健"
        
        tips = HealthTipGenerator.get_tips(category)
        
        if tips:
            tip_text = "\n".join([f"✅ {tip['title']}: {tip['content']}" for tip in tips])
            reply = call_ollama(f"""
            根据以下健康建议，结合用户输入生成友好的回复：
            用户输入: {user_input}
            健康建议: {tip_text}
            用自然、友好的语言表达，不要使用markdown格式
            """)
            return {"success": True, "reply": reply.strip() if reply else tip_text, "tips": tips}
        else:
            return {"success": True, "reply": "根据您的情况，建议保持均衡饮食、规律作息和适度运动，如有需要请咨询专业医生。", "tips": []}

# --------------------------
# 4. 知识库检索 + 风险评估
# --------------------------
def query_knowledge(symptoms=None, disease=None):
    """从医疗知识库中检索匹配信息"""
    conn = sqlite3.connect('medical.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if disease:
        c.execute('SELECT * FROM diseases WHERE name LIKE ?', (f'%{disease}%',))
    elif symptoms:
        c.execute('SELECT * FROM diseases WHERE symptoms LIKE ?', (f'%{symptoms}%',))
    result = [dict(row) for row in c.fetchall()]
    conn.close()
    return result

def risk_assessment(risk_level):
    """根据风险等级生成评估提示"""
    if risk_level == "高":
        return "⚠️ 高风险提示：您描述的症状可能属于急症，请立即停止活动，拨打120或前往最近的急诊就医！"
    elif risk_level == "中":
        return "⚠️ 中风险提示：建议您尽快前往医院就诊，避免延误病情。"
    else:
        return "ℹ️ 低风险提示：您的症状多为常见轻症，可先居家护理观察，若加重请及时就医。"

# --------------------------
# 5. 对话历史管理
# --------------------------
def save_chat_history(user_input, bot_reply, intent):
    """保存对话历史到数据库"""
    try:
        conn = sqlite3.connect('medical.db')
        c = conn.cursor()
        c.execute('INSERT INTO chat_history (user_input, bot_reply, intent) VALUES (?, ?, ?)',
                  (user_input, bot_reply, intent))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"保存对话历史失败: {e}")

def get_chat_history(limit=20):
    """获取对话历史"""
    conn = sqlite3.connect('medical.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM chat_history ORDER BY created_at DESC LIMIT ?', (limit,))
    result = [dict(row) for row in c.fetchall()]
    conn.close()
    return result

def clear_chat_history():
    """清空对话历史"""
    conn = sqlite3.connect('medical.db')
    c = conn.cursor()
    c.execute('DELETE FROM chat_history')
    conn.commit()
    conn.close()

# --------------------------
# 6. 用户认证模块
# --------------------------
def hash_password(password):
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    """验证密码"""
    return hash_password(password) == password_hash

def create_token():
    """创建会话令牌"""
    return secrets.token_hex(16)

@app.route('/api/auth/register', methods=['POST'])
def register():
    """用户注册"""
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400

    if len(username) < 3 or len(password) < 6:
        return jsonify({"error": "用户名至少3位，密码至少6位"}), 400

    try:
        conn = sqlite3.connect('medical.db')
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE username = ?', (username,))
        if c.fetchone():
            conn.close()
            return jsonify({"error": "用户名已存在"}), 409

        token = create_token()
        password_hash = hash_password(password)
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                  (username, password_hash))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "注册成功", "token": token, "username": username})
    except Exception as e:
        return jsonify({"error": f"注册失败: {str(e)}"}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """用户登录"""
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400

    try:
        conn = sqlite3.connect('medical.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()

        if not user or not verify_password(password, user['password_hash']):
            return jsonify({"error": "用户名或密码错误"}), 401

        token = create_token()
        return jsonify({
            "success": True,
            "message": "登录成功",
            "token": token,
            "username": username
        })
    except Exception as e:
        return jsonify({"error": f"登录失败: {str(e)}"}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """用户登出"""
    return jsonify({"success": True, "message": "已退出登录"})

@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    """检查登录状态"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    username = request.headers.get('X-Username', '')
    return jsonify({
        "logged_in": bool(token and username),
        "username": username if token else ""
    })

# --------------------------
# 6. API网关：对话管理接口
# --------------------------
@app.route('/api/chat', methods=['POST'])
def chat():
    start_time = time.time()
    data = request.get_json()
    user_input = data.get("message", "").strip()
    
    if not user_input:
        return jsonify({"reply": "请输入您的问题，我将为您提供医疗咨询服务。", "error": "empty_input"}), 400

    # 检查Ollama服务状态
    if not check_ollama_status():
        error_msg = "❌ 服务连接失败！\n\n请检查：\n1. Ollama服务是否已启动（运行命令: ollama serve）\n2. gemma3:12b模型是否已下载（运行命令: ollama pull gemma3:12b）\n3. 网络连接是否正常"
        return jsonify({
            "reply": error_msg,
            "error": "ollama_not_running",
            "service_available": False
        }), 503

    try:
        # 步骤1：意图识别
        intent = intent_recognition(user_input)
        print(f"识别意图: {intent}")

        # 步骤2：处理工具调用意图
        if intent == "tool_bmi":
            bmi_result = BMICalculator.execute(user_input)
            if bmi_result["success"]:
                reply = f"您的BMI指数为 {bmi_result['bmi']}，属于{bmi_result['category']}范围。\n\n{bmi_result['suggestion']}\n\n{bmi_result['range_info']}"
            else:
                reply = bmi_result.get("error", "BMI计算失败，请检查输入格式。")
            save_chat_history(user_input, reply, intent)
            return jsonify({
                "intent": intent,
                "slots": {},
                "knowledge": [],
                "risk_tip": "",
                "reply": reply,
                "tool_result": bmi_result,
                "service_available": True
            })
        
        if intent == "tool_health_tip":
            tip_result = HealthTipGenerator.generate_personalized_tip(user_input)
            save_chat_history(user_input, tip_result["reply"], intent)
            return jsonify({
                "intent": intent,
                "slots": {},
                "knowledge": tip_result.get("tips", []),
                "risk_tip": "",
                "reply": tip_result["reply"],
                "tool_result": tip_result,
                "service_available": True
            })

        # 步骤3：槽位填充（非工具调用意图）
        slots = slot_filling(user_input, intent)
        print(f"提取槽位: {slots}")

        # 步骤4：知识库检索
        knowledge = []
        if intent == "symptom_consult" and "symptoms" in slots:
            knowledge = query_knowledge(symptoms=slots["symptoms"])
        elif intent == "disease_query" and "disease" in slots:
            knowledge = query_knowledge(disease=slots["disease"])

        # 步骤5：风险评估
        risk_tip = ""
        if knowledge:
            risk_tip = risk_assessment(knowledge[0]["risk_level"])

        # 步骤6：生成回复（结合知识库+模型）
        prompt = f"""
        你是一个专业的医疗咨询助手，根据以下信息回答用户问题：
        用户输入: {user_input}
        知识库信息: {knowledge}
        风险提示: {risk_tip}
        请用通俗易懂的语言回复用户，不要给出诊断结论，只提供建议和风险提示。
        """
        model_reply = call_ollama(prompt)
        
        if not model_reply:
            # 如果模型调用失败，使用知识库信息直接回复
            if knowledge:
                model_reply = f"根据知识库信息，关于{knowledge[0]['name']}的建议：\n\n症状：{knowledge[0]['symptoms']}\n\n建议：{knowledge[0]['suggestion']}\n\n{risk_tip}"
            else:
                model_reply = "抱歉，当前模型服务繁忙，请稍后再试。您也可以尝试描述更具体的症状。"

        save_chat_history(user_input, model_reply, intent)

        end_time = time.time()
        print(f"对话处理耗时: {end_time - start_time:.2f}s")

        return jsonify({
            "intent": intent,
            "slots": slots,
            "knowledge": knowledge,
            "risk_tip": risk_tip,
            "reply": model_reply,
            "service_available": True
        })

    except Exception as e:
        print(f"对话处理异常: {e}")
        return jsonify({
            "reply": f"❌ 服务处理异常：{str(e)}\n\n请稍后重试，或检查Ollama服务状态。",
            "error": str(e),
            "service_available": False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    ollama_status = check_ollama_status()
    return jsonify({
        "service": "medical-chatbot",
        "status": "running",
        "ollama_available": ollama_status,
        "model": MODEL_NAME
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    """获取对话历史"""
    limit = request.args.get('limit', 20)
    history = get_chat_history(int(limit))
    return jsonify({"history": history})

@app.route('/api/history/clear', methods=['DELETE'])
def clear_history():
    """清空对话历史"""
    clear_chat_history()
    return jsonify({"success": True, "message": "对话历史已清空"})

@app.route('/api/tools/bmi', methods=['POST'])
def calculate_bmi():
    """BMI计算API接口"""
    data = request.get_json()
    height = data.get("height")
    weight = data.get("weight")
    
    if not height or not weight:
        return jsonify({"error": "请提供身高和体重"}), 400
    
    result = BMICalculator.calculate(height, weight)
    return jsonify(result)

@app.route('/api/tools/health_tips', methods=['GET'])
def get_health_tips():
    """获取健康建议API接口"""
    category = request.args.get("category")
    tips = HealthTipGenerator.get_tips(category)
    return jsonify({"tips": tips})

@app.route('/api/difficult_diseases', methods=['GET'])
def get_difficult_diseases():
    """获取疑难杂症分类信息"""
    difficult_diseases = [
        {
            "category": "自身免疫性疾病",
            "items": [
                {"name": "系统性红斑狼疮", "description": "症状多样，可能涉及皮肤、关节、肾脏等多器官", "severity": "高"},
                {"name": "类风湿性关节炎", "description": "慢性关节炎症，伴有疼痛和功能障碍", "severity": "中"},
                {"name": "干燥综合征", "description": "主要侵犯外分泌腺，导致口干、眼干", "severity": "中"}
            ]
        },
        {
            "category": "神经系统疾病",
            "items": [
                {"name": "帕金森病", "description": "运动功能障碍，震颤、僵直、行动迟缓", "severity": "高"},
                {"name": "阿尔茨海默病", "description": "进行性认知障碍，记忆力减退", "severity": "高"},
                {"name": "多发性硬化", "description": "中枢神经系统脱髓鞘疾病", "severity": "高"}
            ]
        },
        {
            "category": "遗传代谢疾病",
            "items": [
                {"name": "苯丙酮尿症", "description": "氨基酸代谢障碍，影响神经系统发育", "severity": "高"},
                {"name": "痛风", "description": "尿酸代谢异常，导致关节炎症", "severity": "中"},
                {"name": "肝豆状核变性", "description": "铜代谢障碍，影响肝和神经系统", "severity": "高"}
            ]
        },
        {
            "category": "血液系统疾病",
            "items": [
                {"name": "再生障碍性贫血", "description": "骨髓造血功能衰竭", "severity": "高"},
                {"name": "地中海贫血", "description": "遗传性血红蛋白合成障碍", "severity": "中"},
                {"name": "血友病", "description": "凝血因子缺乏，导致出血倾向", "severity": "高"}
            ]
        },
        {
            "category": "肿瘤相关",
            "items": [
                {"name": "淋巴瘤", "description": "淋巴系统恶性肿瘤", "severity": "高"},
                {"name": "白血病", "description": "造血干细胞恶性克隆性疾病", "severity": "高"},
                {"name": "骨髓瘤", "description": "浆细胞恶性肿瘤", "severity": "高"}
            ]
        }
    ]
    return jsonify({"diseases": difficult_diseases})

@app.route('/api/qa_examples', methods=['GET'])
def get_qa_examples():
    """获取问答示例数据"""
    qa_examples = [
        {"question": "发烧应该怎么处理？", "answer": "发烧时建议多喝水、适当休息，体温超过38.5℃可服用退热药物。若持续高烧或伴有其他严重症状应及时就医。"},
        {"question": "高血压患者饮食应注意什么？", "answer": "高血压患者应减少钠盐摄入，控制脂肪摄入，多吃蔬菜水果，限制饮酒，保持规律运动。"},
        {"question": "如何预防感冒？", "answer": "预防感冒应勤洗手、保持室内通风、适度锻炼、保证充足睡眠、均衡饮食、避免过度劳累和接触感冒患者。"},
        {"question": "糖尿病患者能吃水果吗？", "answer": "糖尿病患者可以适量吃水果，选择低糖水果如苹果、梨、橙子等，在两餐之间食用，并注意控制总量。"},
        {"question": "头痛应该看什么科？", "answer": "一般头痛可先看神经内科或内科。如果有外伤史或特殊症状，应根据情况选择相应专科。"},
        {"question": "体检发现脂肪肝怎么办？", "answer": "脂肪肝患者应控制饮食、减少油腻食物、戒酒、适度运动、控制体重，定期复查肝功能和超声。"}
    ]
    return jsonify({"qa_examples": qa_examples})

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/chat')
def chat_page():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)