import os
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import json

with open("config.json", "r") as f:
    config = json.load(f)
    
api_key = config["OPENAI_API_KEY"]

OPENAI_MODEL = "gpt-o3"

app = FastAPI(title="Advanced OpenAI Gateway Service")
openai_client = AsyncOpenAI(api_key=api_key)

# --- Модели данных ---

class RAGRequest(BaseModel):
    question: str
    context: str

class AnswerResponse(BaseModel):
    answer: str

# --- Эндпоинты API ---

@app.post("/generate_answer", response_model=AnswerResponse)
async def generate_answer(request: RAGRequest):
    """
    Принимает вопрос и контекст, асинхронно обращается к OpenAI и возвращает ответ.
    """
    user_prompt = f"КОНТЕКСТ:\n---\n{request.context}\n---\nВОПРОС: {request.question}\n\nОТВЕТ:"

    try:
        response = await openai_client.chat.completions.create(
            model=config.get("openai_model", "gpt-4o"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.get("temperature", 0.1),
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Произошла ошибка при обращении к OpenAI: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера. Не удалось обработать запрос.")

# --- Запуск приложения ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)