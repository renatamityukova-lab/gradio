"""Формирование запроса к LLM для сценария «список покупок по блюду».

Количество человек и формат ответа передаются в промпт явными инструкциями,
а не подразумеваются. Модуль не выполняет сетевых вызовов.
"""

from __future__ import annotations

FORMAT_PRODUCTS = "Список продуктов"
FORMAT_WITH_STEPS = "Список + шаги"
FORMAT_CHOICES = [FORMAT_PRODUCTS, FORMAT_WITH_STEPS]
DEFAULT_FORMAT = FORMAT_PRODUCTS

PEOPLE_CHOICES = ["1", "2", "4", "6"]
DEFAULT_PEOPLE = "2"

SYSTEM_PROMPT = (
    "Ты помощник по планированию покупок для приготовления блюд. "
    "Отвечай на русском языке, без вступлений и рассуждений о себе. "
    "Строго соблюдай количество человек и формат ответа, заданные пользователем."
)

# Явные инструкции по формату: модель не выбирает формат сама.
_FORMAT_INSTRUCTIONS = {
    FORMAT_PRODUCTS: (
        "Формат ответа: ТОЛЬКО список покупок. "
        "Каждый продукт — отдельной строкой, начиная с «- ». "
        "Одна строка — один продукт, без объединения нескольких продуктов в строку. "
        "Не добавляй шаги приготовления, вступления и итоговые выводы."
    ),
    FORMAT_WITH_STEPS: (
        "Формат ответа: сначала список покупок, затем шаги приготовления. "
        "Блок «Список покупок» — каждый продукт отдельной строкой с «- », "
        "одна строка — один продукт. "
        "Затем блок «Шаги приготовления» — от 5 до 8 коротких нумерованных пунктов. "
        "Шаги должны быть краткими, по одному действию на пункт."
    ),
}


def normalize_people(people: str) -> str:
    """Привести количество человек к допустимому значению."""
    value = (people or "").strip()
    return value if value in PEOPLE_CHOICES else DEFAULT_PEOPLE


def normalize_format(response_format: str) -> str:
    """Привести формат ответа к допустимому значению."""
    return response_format if response_format in _FORMAT_INSTRUCTIONS else DEFAULT_FORMAT


def build_user_prompt(dish: str, people: str, format: str) -> str:
    """Собрать пользовательский промпт с явными количеством человек и форматом."""
    dish = (dish or "").strip()
    people = normalize_people(people)
    response_format = normalize_format(format)

    return "\n".join(
        [
            f"Блюдо: {dish}",
            f"Количество человек: {people}",
            _FORMAT_INSTRUCTIONS[response_format],
            "",
            f"Рассчитай продукты и их количество на {people} чел.",
            "Количества продуктов должны соответствовать указанному числу человек.",
        ]
    )


def build_messages(dish: str, people: str, format: str) -> list[dict[str, str]]:
    """Собрать сообщения для chat/completions: system + user."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(dish, people, format)},
    ]
