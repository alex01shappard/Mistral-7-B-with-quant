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
## Использование квантизации
Для оптимизации использования ресурсов (особенно на GPU) в проекте применяется квантизация модели. Используется конфигурация `BitsAndBytesConfig`, которая позволяет загружать модель в 4-битном формате. Это уменьшает объём памяти, необходимый для работы модели, без значительных потерь в точности.

Пример настройки квантизации:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    device_map="auto",
    quantization_config=quantization_config,
)
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



## Важные замечания
- Код работает как на CPU, так и на GPU. Убедитесь, что у вас есть поддержка CUDA для ускорения работы модели.
- Шаблон для запросов настроен для обеспечения вежливого и точного взаимодействия с моделью.
