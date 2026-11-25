"""Клиент LLM API для проекта gradio.

Единственное место сетевого вызова модели — UI-код его не дублирует.
Настройки берутся из переменных окружения (``gradio/.env``).
Ключ не пишется в логи и не попадает в тексты ошибок.

Карта ошибок:
* 4xx (доступ/настройка) — без повторов;
* 5xx и 429 — максимум один повтор;
* таймаут/сеть — максимум один повтор;
* пустой или битый ответ — без повторов.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).resolve().parent / ".env"
DEFAULT_BASE_URL = "https://llm.enplus.group/v1"
DEFAULT_MODEL = "gpt-oss-120b"


class LLMError(Exception):
    """Базовая ошибка LLM.

    ``user_message`` безопасен для показа пользователю: без ключа и деталей.
    """

    user_message = "Не удалось получить ответ. Попробуйте ещё раз."


class LLMAccessError(LLMError):
    """4xx: доступ или настройка. Повторы не выполняются."""

    user_message = "Не настроен доступ к сервису."


class LLMServiceUnavailableError(LLMError):
    """5xx или 429: сервис временно недоступен. Допускается один повтор."""

    user_message = "Сервис временно недоступен. Попробуйте позже."


class LLMNetworkError(LLMError):
    """Таймаут или сетевая ошибка. Допускается один повтор."""

    user_message = "Не удалось получить ответ. Попробуйте ещё раз."


class LLMEmptyResponseError(LLMError):
    """Пустой или невалидный ответ модели. Повторы не выполняются."""

    user_message = "Пустой ответ модели. Попробуйте переформулировать вопрос."


# Ошибки, при которых запрос имеет смысл повторить один раз.
_RETRYABLE_ERRORS = (LLMServiceUnavailableError, LLMNetworkError)


@dataclass(frozen=True)
class LLMConfig:
    """Параметры подключения к LLM."""

    model: str
    base_url: str
    api_key: str
    fallback_model: str | None = None
    fallback_base_url: str | None = None
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    max_retries: int = 1
    trust_env: bool = True


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def load_config() -> LLMConfig:
    """Собрать конфигурацию из окружения.

    :raises LLMAccessError: не задан ключ доступа.
    """
    load_dotenv(ENV_PATH, override=False)

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise LLMAccessError(
            "LLM не настроена: заполните OPENAI_API_KEY в gradio/.env"
        )

    base_url = (os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
    model = (os.getenv("LLM_MODEL") or DEFAULT_MODEL).strip()
    fallback_model = (os.getenv("LLM_FALLBACK_MODEL") or "").strip() or None
    fallback_base_url = (
        os.getenv("LLM_FALLBACK_BASE_URL") or ""
    ).strip().rstrip("/") or None

    return LLMConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        fallback_model=fallback_model,
        fallback_base_url=fallback_base_url,
        temperature=_as_float(os.getenv("LLM_TEMPERATURE"), 0.0),
        timeout_seconds=_as_float(os.getenv("LLM_TIMEOUT_SECONDS"), 60.0),
        # По умолчанию ровно один повтор для временных ошибок.
        max_retries=_as_int(os.getenv("LLM_MAX_RETRIES"), 1),
        trust_env=_as_bool(os.getenv("LLM_TRUST_ENV"), True),
    )


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _body_snippet(response: httpx.Response, limit: int = 200) -> str:
    try:
        return response.text[:limit].replace("\n", " ")
    except Exception:  # noqa: BLE001 — тело может быть недоступно
        return "<тело недоступно>"


def _post_once(
    client: httpx.Client,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    """Один запрос к модели без повторов. Ошибки — типизированные."""
    payload = {"model": model, "temperature": temperature, "messages": messages}

    logger.info("LLM: запрос — model=%s, POST %s", model, url)

    try:
        response = client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        logger.warning("LLM: таймаут — model=%s", model)
        raise LLMNetworkError(f"{model}: таймаут запроса") from exc
    except httpx.TransportError as exc:
        logger.warning("LLM: сетевая ошибка — model=%s, %s", model, type(exc).__name__)
        raise LLMNetworkError(
            f"{model}: сетевая ошибка ({type(exc).__name__})"
        ) from exc

    status = response.status_code

    if status == 429 or status >= 500:
        logger.warning("LLM: временная ошибка HTTP %s — model=%s", status, model)
        raise LLMServiceUnavailableError(f"{model}: временная ошибка HTTP {status}")

    if status >= 400:
        # 4xx: доступ/настройка или отклонённый запрос — без повторов.
        logger.error(
            "LLM: запрос отклонён HTTP %s — model=%s: %s",
            status,
            model,
            _body_snippet(response),
        )
        raise LLMAccessError(
            f"{model}: запрос отклонён (HTTP {status}): {_body_snippet(response)}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        logger.error("LLM: ответ не является JSON — model=%s", model)
        raise LLMEmptyResponseError(f"{model}: ответ не является JSON") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.error("LLM: неожиданный формат ответа — model=%s", model)
        raise LLMEmptyResponseError(f"{model}: неожиданный формат ответа") from exc

    if not isinstance(content, str) or not content.strip():
        logger.error("LLM: пустой ответ — model=%s", model)
        raise LLMEmptyResponseError(f"{model}: модель вернула пустой ответ")

    logger.info("LLM: ответ получен — model=%s, HTTP %s", model, status)
    return content.strip()


def _complete_model(
    client: httpx.Client,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    attempts: int,
) -> str:
    """Запрос к одной модели: повторы только для временных ошибок."""
    last_error: LLMError | None = None

    for attempt in range(attempts):
        try:
            return _post_once(client, url, model, messages, temperature)
        except _RETRYABLE_ERRORS as exc:
            last_error = exc
            if attempt + 1 < attempts:
                delay = min(2**attempt, 5)
                logger.warning(
                    "LLM: попытка %d/%d не удалась (%s) — повтор через %d с",
                    attempt + 1,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)

    if last_error is None:  # защита от неконсистентного состояния
        raise LLMError(f"{model}: запрос не выполнен")
    raise type(last_error)(
        f"{model}: недоступна после {attempts} попыток ({last_error})"
    ) from last_error


def complete(
    messages: list[dict[str, str]],
    *,
    config: LLMConfig | None = None,
) -> str:
    """Отправить сообщения в LLM и вернуть текст ответа.

    Сначала используется основная модель, затем ``fallback_model`` (если задана).
    Для 4xx резервная модель не пробуется: повтор не поможет.

    :raises LLMAccessError: не настроен ключ или сервис отклонил запрос (4xx).
    :raises LLMServiceUnavailableError: сервис недоступен (5xx/429).
    :raises LLMNetworkError: таймаут или сетевая ошибка.
    :raises LLMEmptyResponseError: модель вернула пустой или битый ответ.
    """
    cfg = config or load_config()

    targets: list[tuple[str, str]] = [
        (cfg.model, _chat_completions_url(cfg.base_url))
    ]
    if cfg.fallback_model:
        targets.append(
            (
                cfg.fallback_model,
                _chat_completions_url(cfg.fallback_base_url or cfg.base_url),
            )
        )

    attempts = max(1, cfg.max_retries + 1)
    errors: list[LLMError] = []

    with httpx.Client(
        timeout=cfg.timeout_seconds,
        trust_env=cfg.trust_env,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
    ) as client:
        for index, (model, url) in enumerate(targets, start=1):
            if index > 1:
                logger.warning("LLM: переключение на резервную модель %s", model)
            try:
                return _complete_model(
                    client,
                    url,
                    model,
                    messages,
                    cfg.temperature,
                    attempts,
                )
            except LLMAccessError as exc:
                # 4xx — настройка/доступ: повтор и резервная модель не помогут.
                errors.append(exc)
                logger.error("LLM: доступ отклонён, повторы прекращены: %s", exc)
                break
            except LLMError as exc:
                errors.append(exc)
                logger.warning("LLM: модель %s не дала результат: %s", model, exc)

    summary = "; ".join(str(error) for error in errors)
    logger.error("LLM: получить ответ не удалось — %s", summary)

    # Сохраняем категорию ошибки, чтобы UI показал нужное сообщение.
    raise type(errors[-1])(summary) from errors[-1]
