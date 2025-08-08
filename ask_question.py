import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import json

with open("config.json", "r") as f:
    config = json.load(f)
    
api_key = config["OPENAI_API_KEY"]

OPENAI_MODEL = "gpt-o3"

app = FastAPI(title="OpenAI Gateway Service")
openai_client = OpenAI(api_key=api_key)

class RAGRequest(BaseModel):
    question: str
    context: str

@app.post("/generate_answer")
async def generate_answer(request: RAGRequest):
    """
    Принимает вопрос и контекст, обращается к OpenAI и возвращает ответ.
    """
    system_prompt = """
    Перефразируй промпт на основе контекста базы знаний.

    Ты — эксперт-аналитик, работающий с базой знаний компании. Твоя задача — предоставить ответ на вопрос пользователя, основываясь на предоставленных фрагментах документов (контексте) и твоей интерпретации их.
    Внимательно изучи весь контекст. Синтезируй информацию из разных фрагментов, если это необходимо для полноты ответа.
    Если после анализа ты уверен, что в контексте нет ответа на вопрос, вежливо сообщи: 'К сожалению, в предоставленных документах не найдено информации по вашему вопросу.'
    """
    user_prompt = f"КОНТЕКСТ:\n---\n{request.context}\n---\nВОПРОС: {request.question}\n\nОТВЕТ:"

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)