import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import google.generativeai as genai
import json

load_dotenv()

# Gemini 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini 모델 설정
# 할당량 이슈가 있는 Search Grounding(Tool) 기능을 제거하고, 
# 모델의 내부 학습 데이터를 활용하여 실용/창의 레시피를 구분하도록 설정합니다.
model = genai.GenerativeModel('gemini-3-flash-preview')

app = FastAPI()

# CORS 설정 (모바일 PWA 접속 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/suggest-recipe")
async def suggest_recipe(data: dict):
    try:
        items = data.get("items", [])
        if not items:
            raise HTTPException(status_code=400, detail="식재료 목록이 비어 있습니다.")
            
        items_str = ", ".join([f"{item['name']}({item['quantity']})" for item in items])
        
        prompt = f"""
        내가 지금 가진 식재료 목록이야: {items_str}
        
        이 재료들을 바탕으로 세 가지 타입의 레시피를 각각 최대 3개씩 제안해줘. (재료가 부족하면 1~2개도 괜찮아):
        
        1. '실용 레시피(practical)': 반드시 인터넷이나 요리책 등에 실제로 존재하는(Real-world) 유명하거나 검증된 레시피여야 해. 현재 가진 재료로 만들 수 있는 실제 요리 이름을 찾아서 제안해줘.
        2. '창작 레시피(creative)': AI인 너의 창의성을 발휘하여, 기존에 없던 독특하고 화려한 인플루언서용 새로운 요리를 직접 '발명'해줘. 이 요리는 인터넷에 없을수록 좋아.
        3. '10분 레시피(quick)': 자취생이나 직장인을 위해, 최소한의 재료와 단계로 10분 내에 뚝딱 만들 수 있는 초간단 레시피를 제안해줘. 
        
        결과는 반드시 다음과 같은 JSON 형식으로만 응답해줘 (반드시 리스트 형식이어야 함):
        {{
          "practical": [
            {{
              "recipe_name": "실제 요리 이름 1",
              "ingredients": "필요 재료",
              "instructions": ["단계 1", "단계 2", ...],
              "caption": "현실적인 SNS 캡션"
            }},
            ...
          ],
          "creative": [
            {{
              "recipe_name": "창작 요리 이름 1",
              "ingredients": "필요 재료 (추가 필요 재료 포함 가능)",
              "instructions": ["단계 1", "단계 2", ...],
              "caption": "화려한 인플루언서용 SNS 캡션"
            }},
            ...
          ],
          "quick": [
            {{
              "recipe_name": "10분 요리 이름 1",
              "ingredients": "필수 재료",
              "instructions": ["단계 1", "단계 2", ...],
              "caption": "빠르고 실용적인 SNS 캡션"
            }},
            ...
          ]
        }}
        """
        
        # Gemini 분석 요청 (모델의 내부 지식 기반으로 실용/창의 분리)
        print(f"DEBUG: Suggesting dual recipes using internal knowledge on model {model.model_name}")
        response = model.generate_content(prompt)
        
        # JSON 형식만 추출
        raw_text = response.text
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
        result = json.loads(raw_text)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# API 엔드포인트는 정적 파일 서빙보다 먼저 정의되어야 합니다.
@app.post("/analyze-receipt")
async def analyze_receipt(file: UploadFile = File(...)):
    try:
        # 파일 읽기
        content = await file.read()
        
        # prompt 정의 (식재료 추출 최적화)
        prompt = """
        이 영수증 이미지에서 구매한 식재료(식자재) 품목과 수량(또는 무게)을 정확히 추출해줘.
        
        주의사항:
        1. 반드시 '먹을 수 있는' 식재료만 포함해. 
        2. 주방용품(종이컵, 위생장갑), 생필품(세제, 휴지), 의류, 잡화 등 식재료가 아닌 항목은 절대 포함하지 마.
        3. 품목명은 이해하기 쉽게 다듬어서 응답해줘. (예: '국내산 냉장 삼겹살 500g' -> '삼겹살')
        
        결과는 반드시 다음과 같은 JSON 형식으로만 응답해줘:
        {
          "items": [
            {"name": "품목명", "quantity": "수량/무게"},
            ...
          ],
          "total_price": "총액",
          "date": "구매날짜"
        }
        """
        
        # Gemini 분석 요청 (안정적인 분석 수행)
        print(f"DEBUG: Analyzing receipt using model {model.model_name}")
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": content}
        ])
        
        # JSON 형식만 추출 (마크다운태그 제거 등)
        raw_text = response.text
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
        result = json.loads(raw_text)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 프론트엔드 정적 파일 서빙 (모든 API 루트 정의 후 마지막에 배치)
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "../frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
