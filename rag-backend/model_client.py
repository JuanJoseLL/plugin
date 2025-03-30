import time
from typing import List, Optional # Import Optional
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
# Remove unused imports if ChatResult, ChatGeneration aren't needed directly
# from langchain_core.outputs import ChatResult, ChatGeneration
import requests
from config import (
    QWEN_API_URL,
    API_KEY,
    MODEL_NAME,
    LLM_TIMEOUT,
    LLM_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_REPETITION_PENALTY,get_logger)
logger = get_logger(__name__)
# Import CallbackManagerForLLMRun for type hinting (optional but good practice)
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class CustomChatQwen(SimpleChatModel):
    """Wraps the custom Qwen API endpoint."""
    model_name: str = MODEL_NAME
    api_key: str = API_KEY
    api_url: str = QWEN_API_URL
    max_tokens: int = LLM_MAX_TOKENS
    temperature: float = LLM_TEMPERATURE
    top_k: int = LLM_TOP_K
    repetition_penalty: float = LLM_REPETITION_PENALTY

    @property
    def _llm_type(self) -> str:
        return "custom_chat_qwen"

    # Add run_manager type hint using Optional for clarity
    def _call(self, messages: List[BaseMessage], stop: List[str] | None = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """Makes the API call to the Qwen endpoint."""
        # Format messages for the API
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                 api_messages.append({"role": "assistant", "content": msg.content})
            # Add other message types if needed (ToolMessage, etc.)

        # --- START FIX ---
        # Build the payload ONLY with parameters expected by the Qwen API.
        # Explicitly exclude internal LangChain objects like run_manager.
        payload = {
            "model": self.model_name,
            "messages": api_messages,
            # Use defaults unless overridden in kwargs that ARE NOT run_manager
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_k": kwargs.get("top_k", self.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
        }

        # Add 'stop' sequences if provided and if the API supports it
        # Check Qwen API documentation for the correct parameter name (e.g., 'stop', 'stop_sequences')
        if stop:
             payload["stop"] = stop # Adjust key name 'stop' if needed based on Qwen API docs

        # Optionally, add any *other* kwargs that you know the Qwen API accepts
        # Be careful not to blindly add all kwargs
        # Example: if Qwen supports a 'seed' parameter passed via kwargs:
        # if "seed" in kwargs:
        #     payload["seed"] = kwargs["seed"]

        # --- END FIX ---


        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # --- Retry logic remains the same ---
        for attempt in range(LLM_RETRIES):
            try:
                # Make sure to pass the *cleaned* payload
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=LLM_TIMEOUT)

                # Trigger callback manager's error handling *if* run_manager is present
                # This part is optional but good practice if using LangChain's callbacks
                try:
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    if run_manager:
                         run_manager.on_llm_error(e, response=response)
                    raise e # Re-raise the exception to be caught by the outer try-except

                data = response.json()
                if data.get("choices") and data["choices"][0].get("message") and data["choices"][0]["message"].get("content"):
                    content = data["choices"][0]["message"]["content"]
                    # Trigger callback manager's success handling *if* run_manager is present
                    if run_manager:
                        # Construct the necessary arguments for on_llm_new_token if needed,
                        # or simplify if just logging the final result.
                        # For simplicity here, we'll assume the callback system handles the final result.
                        pass # Callbacks are often handled by the calling LangChain framework itself
                    return content
                else:
                    error_msg = f"Invalid response structure from LLM API: {data}"
                    logger.error(error_msg)
                    if run_manager:
                        run_manager.on_llm_error(ValueError(error_msg), response=response)
                    raise ValueError("Invalid LLM API response structure.")
            except requests.exceptions.Timeout as e:
                 logger.error(f"LLM request timed out (attempt {attempt+1}/{LLM_RETRIES}).")
                 if run_manager:
                    run_manager.on_llm_error(e)
                 if attempt + 1 == LLM_RETRIES: raise TimeoutError("LLM request timed out.")
                 time.sleep(2 * (attempt + 1))
            except requests.exceptions.RequestException as e:
                logger.error(f"LLM request failed (attempt {attempt+1}/{LLM_RETRIES}): {e}")
                # run_manager.on_llm_error already called above if raise_for_status failed
                if attempt + 1 == LLM_RETRIES: raise ConnectionError(f"LLM request failed: {e}")
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                 # Catch the JSONDecodeError specifically if needed
                 # except (json.JSONDecodeError, ValueError, Exception) as e:
                 logger.error(f"Unexpected error during LLM call (attempt {attempt+1}/{LLM_RETRIES}): {e}")
                 if run_manager:
                    run_manager.on_llm_error(e)
                 if attempt + 1 == LLM_RETRIES: raise RuntimeError(f"Unexpected LLM error: {e}")
                 time.sleep(2 * (attempt + 1))

        # Should not be reached if retries > 0
        raise RuntimeError("Failed to get LLM response after multiple retries.")