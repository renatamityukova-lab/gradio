"""Gradio-интерфейс «Список покупок по блюду».

Поток: валидация ввода -> формирование промпта -> вызов LLM -> текст результата.
Сетевой вызов инкапсулирован в модуле ``llm_client.py``.
"""

from __future__ import annotations

import gradio as gr

try:  # запуск как скрипта: python gradio/gradio_app.py
    from prompts import (
        DEFAULT_FORMAT,
        DEFAULT_PEOPLE,
        FORMAT_CHOICES,
        PEOPLE_CHOICES,
        build_messages,
    )
except ImportError:  # импорт как пакета
    from .prompts import (
        DEFAULT_FORMAT,
        DEFAULT_PEOPLE,
        FORMAT_CHOICES,
        PEOPLE_CHOICES,
        build_messages,
    )

try:  # запуск как скрипта: python gradio/gradio_app.py
    from validation import MAX_DISH_LENGTH, validate_dish
except ImportError:  # импорт как пакета
    from .validation import MAX_DISH_LENGTH, validate_dish

try:  # запуск как скрипта: python gradio/gradio_app.py
    from llm_client import LLMError, complete
except ImportError:  # импорт как пакета
    from .llm_client import LLMError, complete

TITLE = "Список покупок по блюду"
DESCRIPTION = (
    "Укажите блюдо, количество человек и формат ответа — "
    "приложение подготовит список покупок."
)


def generate_shopping_list(dish: str, people: str, format: str) -> str:
    """Обработчик кнопки: валидация -> промпт -> вызов LLM -> результат.

    Контракт: ``dish: str, people: str, format: str`` -> строка результата.
    """
    error = validate_dish(dish)
    if error:
        return error

    messages = build_messages(dish, people, format)
    try:
        return complete(messages)
    except LLMError as exc:
        # Пользователю — только безопасное сообщение по карте ошибок,
        # технические детали остаются в логах llm_client.
        return exc.user_message


def create_app() -> gr.Blocks:
    """Собрать интерфейс: ввод -> обработчик -> вывод."""
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")

        dish_input = gr.Textbox(
            label="Что приготовить",
            placeholder="Например: борщ",
            info=f"Не более {MAX_DISH_LENGTH} символов",
            lines=3,
            max_lines=8,
        )
        people_input = gr.Dropdown(
            label="На сколько человек",
            choices=PEOPLE_CHOICES,
            value=DEFAULT_PEOPLE,
        )
        format_input = gr.Dropdown(
            label="Формат ответа",
            choices=FORMAT_CHOICES,
            value=DEFAULT_FORMAT,
        )
        generate_button = gr.Button(
            "Сгенерировать список покупок",
            variant="primary",
        )
        result_output = gr.Textbox(
            label="Результат",
            lines=16,
            interactive=False,
        )

        generate_button.click(
            fn=generate_shopping_list,
            inputs=[dish_input, people_input, format_input],
            outputs=result_output,
        )

    return demo


demo = create_app()

if __name__ == "__main__":
    demo.launch()
