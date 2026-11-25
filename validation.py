"""Валидация пользовательского ввода до вызова API модели."""

from __future__ import annotations

# Порог выбран с запасом: описание блюда или короткий рецепт помещаются,
# а простыня текста — нет.
MAX_DISH_LENGTH = 500

EMPTY_DISH_MESSAGE = "Напишите, что хотите приготовить."
TOO_LONG_DISH_MESSAGE = "Слишком длинный текст — сократите или разбейте на части."


def validate_dish(dish: str) -> str | None:
    """Проверить описание блюда.

    :return: сообщение об ошибке или ``None``, если ввод корректен.
    """
    text = (dish or "").strip()
    if not text:
        return EMPTY_DISH_MESSAGE
    if len(text) > MAX_DISH_LENGTH:
        return TOO_LONG_DISH_MESSAGE
    return None
