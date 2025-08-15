import os
import json
import logging
import sys
from typing import Dict, Any, List

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

# --- Модели данных (ИЗМЕНЕНЫ) ---

class SourceChunk(BaseModel):
    text: str
    file: str

class RAGRequest(BaseModel):
    question: str
    context: List[SourceChunk] # Контекст теперь - список объектов

class AnswerParagraph(BaseModel):
    paragraph: str
    source: SourceChunk

class AnswerResponse(BaseModel):
    answer: List[AnswerParagraph] # Ответ теперь - список объектов

# --- Эндпоинты API ---

class OpenAIService:
    def __init__(self, config: AppConfig):
        self.config = config
    
    # ИЗМЕНЕНО: Формируем промпт из списка фрагментов
    def _build_user_prompt(self, question: str, context: List[SourceChunk]) -> str:
        context_parts = []
        for i, chunk in enumerate(context):
            context_parts.append(f"ФРАГМЕНТ {i+1} (ИСТОЧНИК: {chunk.file}):\n{chunk.text}")
        
        formatted_context = "\n---\n".join(context_parts)
        return f"КОНТЕКСТ:\n---\n{formatted_context}\n---\nВОПРОС: {question}"
    
    # ИЗМЕНЕНО: Обрабатываем JSON-ответ от модели
    async def generate_answer(self, question: str, context: List[SourceChunk]) -> List[Dict[str, Any]]:
        user_prompt = self._build_user_prompt(question, context)
        
        try:
            response = await self.config.openai_client.chat.completions.create(
                model=self.config.config.get("openai_model", "gpt-4o"),
                response_format={"type": "json_object"}, # Просим модель вернуть JSON
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.config.get("temperature", 0.1),
            )
            raw_answer = response.choices[0].message.content
            
            try:
                # Пытаемся распарсить JSON из ответа модели
                parsed_json = json.loads(raw_answer)
                # Модель может вернуть список или словарь с ключом "answer"
                if isinstance(parsed_json, list):
                    return parsed_json
                elif isinstance(parsed_json, dict) and "answer" in parsed_json and isinstance(parsed_json["answer"], list):
                    return parsed_json["answer"]
                else:
                    logger.warning(f"LLM вернула неожиданную JSON-структуру: {raw_answer}")
                    return []
            except json.JSONDecodeError:
                logger.error(f"Не удалось декодировать JSON от LLM: {raw_answer}")
                return []

        except Exception as e:
            logger.error(f"Ошибка при обращении к OpenAI: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Внутренняя ошибка сервера. Не удалось обработать запрос."
            )

openai_service = OpenAIService(app_config)

@app.post("/generate_answer", response_model=AnswerResponse)
async def generate_answer(request: RAGRequest):
    answer_list = await openai_service.generate_answer(request.question, request.context)
    return {"answer": answer_list}

# --- Запуск приложения ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)