## Описание
Этот проект предназначен для работы с языковой моделью **Mistral-7B-Instruct** через библиотеки `langchain` и `transformers`. Код загружает и настраивает модель для генерации текстов на основе пользовательских запросов.

## Установка
Убедитесь, что у вас установлен Python 3.7 или выше. Затем выполните следующие команды для установки зависимостей:

```bash
pip install -q -U langchain transformers bitsandbytes accelerate huggingface_hub
pip install langchain_community
```

## Настройка
Перед запуском убедитесь, что у вас есть доступ к модели **Mistral-7B-Instruct** через [HuggingFace Hub](https://huggingface.co/). Выполните вход:

```python
from huggingface_hub import login
login()
```

## Пример использования
Код предоставляет функцию `generate_response`, которая генерирует ответ на заданный вопрос.

```python
def generate_response(question):
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run({"question": question})
    return response

response = generate_response("What is space?")
print(response)
```

## Этические ограничения
Перед использованием необходимо внедрить фильтрацию запросов для предотвращения генерации вредоносного контента. Пример:

```python
def is_request_safe(question):
    prohibited_keywords = ["hack", "violence", "illegal"]
    return not any(keyword in question.lower() for keyword in prohibited_keywords)

def generate_response_safe(question):
    if not is_request_safe(question):
        return "This request violates ethical guidelines and cannot be processed."
    return generate_response(question)
```

## Важные замечания
- Код работает как на CPU, так и на GPU. Убедитесь, что у вас есть поддержка CUDA для ускорения работы модели.
- Шаблон для запросов настроен для обеспечения вежливого и точного взаимодействия с моделью.