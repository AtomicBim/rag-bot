import os
import json
import logging
import sys
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- Конфигурация ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AppConfig:
    def __init__(self):
        load_dotenv()
        self.config = self._load_config()
        self.system_prompt = self._load_system_prompt()
        self.openai_client = self._setup_openai_client()
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Файл config.json не найден")
            sys.exit(1)
        except json.JSONDecodeError:
            logger.error("Ошибка чтения JSON из config.json")
            sys.exit(1)
    
    def _load_system_prompt(self) -> str:
        try:
            with open("system_prompt.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error("Файл system_prompt.txt не найден")
            sys.exit(1)
    
    def _setup_openai_client(self) -> AsyncOpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("Переменная окружения OPENAI_API_KEY не установлена")
            raise ValueError("Необходимо установить переменную окружения OPENAI_API_KEY")
        return AsyncOpenAI(api_key=api_key)


app_config = AppConfig()
app = FastAPI(title="Advanced OpenAI Gateway Service")

# --- Модели данных ---

class RAGRequest(BaseModel):
    question: str
    context: str

class AnswerResponse(BaseModel):
    answer: str

# --- Эндпоинты API ---

class OpenAIService:
    def __init__(self, config: AppConfig):
        self.config = config
    
    def _build_user_prompt(self, question: str, context: str) -> str:
        return f"КОНТЕКСТ:\n---\n{context}\n---\nВОПРОС: {question}\n\nОТВЕТ:"
    
    async def generate_answer(self, question: str, context: str) -> str:
        user_prompt = self._build_user_prompt(question, context)
        
        try:
            response = await self.config.openai_client.chat.completions.create(
                model=self.config.config.get("openai_model", "gpt-4o"),
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.config.get("temperature", 0.1),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Ошибка при обращении к OpenAI: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Внутренняя ошибка сервера. Не удалось обработать запрос."
            )


openai_service = OpenAIService(app_config)


@app.post("/generate_answer", response_model=AnswerResponse)
async def generate_answer(request: RAGRequest):
    """
    Принимает вопрос и контекст, асинхронно обращается к OpenAI и возвращает ответ.
    """
    answer = await openai_service.generate_answer(request.question, request.context)
    return {"answer": answer}

# --- Запуск приложения ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)