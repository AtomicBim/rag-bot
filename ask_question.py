import os
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- Конфигурация и инициализация ---

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка переменных окружения из файла .env
load_dotenv()

# Загрузка основной конфигурации
try:
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    logging.error("Файл config.json не найден. Завершение работы.")
    exit()
except json.JSONDecodeError:
    logging.error("Ошибка чтения JSON из config.json. Завершение работы.")
    exit()

# Загрузка системного промпта
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
except FileNotFoundError:
    logging.error("Файл system_prompt.txt не найден. Завершение работы.")
    exit()

# Получение ключа API и настройка клиента OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("Переменная окружения OPENAI_API_KEY не установлена.")
    raise ValueError("Необходимо установить переменную окружения OPENAI_API_KEY")

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