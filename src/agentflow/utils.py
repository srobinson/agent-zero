import inspect
from typing import Any, Callable, Dict, List, Optional, Union


def populate_template(template: Any, data: Dict[str, Any]) -> Any:
    """
    Recursively populates a template with data.

    Args:
        template (Any): The template to populate. Can be a dict, list, or primitive value.
        data (Dict[str, Any]): The data to use for populating the template.

    Returns:
        Any: The populated template.
    """
    if isinstance(template, dict):
        result = {}
        for key, value in template.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                key_in_data = value[1:-1]
                result[key] = data.get(key_in_data, value)
            else:
                result[key] = populate_template(value, data)
        return result
    elif isinstance(template, list):
        return [populate_template(item, data) for item in template]
    else:
        return template


def function_to_json(
    func: Callable, format_template: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Converts a Python function into a JSON-serializable dictionary based on a custom format template.

    Args:
        func (Callable): The function to be converted.
        format_template (Optional[Dict[str, Any]]): A dictionary specifying the desired output structure.
            Use placeholders like '{name}', '{description}', '{parameters}', '{required}'
            as keys or values to indicate where function data should be inserted.
            If None, a default format is used.

    Returns:
        Dict[str, Any]: A dictionary representing the function's signature in the specified format.

    Raises:
        ValueError: If the function signature cannot be obtained.
    """
    # Default type mapping for annotations
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    # Get function signature
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    # Build parameters dynamically
    parameters = {}
    for param in signature.parameters.values():
        param_type = (
            type_map.get(param.annotation, "string")
            if param.annotation != inspect.Parameter.empty
            else "string"
        )
        param_details = {"type": param_type}
        if param.default != inspect.Parameter.empty:
            param_details["default"] = param.default
        parameters[param.name] = param_details

    # Identify required parameters
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect.Parameter.empty
    ]

    # Default format if none provided
    if format_template is None:
        format_template = {
            "type": "function",
            "function": {
                "name": "{name}",
                "description": "{description}",
                "parameters": {
                    "type": "object",
                    "properties": "{parameters}",
                    "required": "{required}",
                    "additionalProperties": False,
                },
            },
            "strict": True,
        }

    # Extract function metadata
    func_data = {
        "name": func.__name__,
        "description": (func.__doc__ or "").strip(),
        "parameters": parameters,
        "required": required if required else [],
    }

    return populate_template(format_template, func_data)


def container_to_json(
    container: Any, format_template: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Converts a Container instance into a JSON-serializable dictionary based on a custom format template.

    Args:
        container (Any): The Container instance to be converted.
        format_template (Optional[Dict[str, Any]]): A dictionary specifying the desired output structure.
            Use placeholders like '{name}', '{description}', '{parameters}', '{required}'
            as keys or values to indicate where container data should be inserted.
            If None, a default format is used.

    Returns:
        Dict[str, Any]: A dictionary representing the container's attributes in the specified format.
    """
    # Default type mapping for annotations
    type_map = {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
        "null": "null",
    }

    # Build parameters dynamically from environment variables
    parameters = {}
    required = []

    for env_var in container.environment:
        param_type = type_map.get(env_var.get("type", "string"), "string")
        param_details = {"type": param_type}
        parameters[env_var["name"]] = param_details
        required.append(env_var["name"])

    # Default format if none provided
    if format_template is None:
        format_template = {
            "type": "container",
            "container": {
                "name": "{name}",
                "description": "{description}",
                "parameters": {
                    "type": "object",
                    "properties": "{parameters}",
                    "required": "{required}",
                    "additionalProperties": False,
                },
            },
            "strict": True,
        }

    # Extract container metadata
    container_data = {
        "name": container.name,
        "description": container.description,
        "parameters": parameters,
        "required": required,
    }

    return populate_template(format_template, container_data)


def extract_key_values(
    tool_call_output: Dict[str, Any], keys_to_find: List[str]
) -> Dict[str, Any]:
    """
    Extracts values for specified keys from a tool_call output dictionary.

    Args:
        tool_call_output (Dict[str, Any]): The dictionary representing the populated tool_call output.
        keys_to_find (List[str]): A list of key names to search for (e.g., ["id", "name", "arguments"]).

    Returns:
        Dict[str, Any]: A dictionary mapping each specified key to its value(s) from the output.
    """
    result = {
        key: [] for key in keys_to_find
    }  # Initialize with empty lists for each key

    # Helper function to recursively search the dictionary
    def search_dict(data: Any, target_keys: List[str]) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                if key in target_keys:
                    result[key].append(value)
                search_dict(value, target_keys)
        elif isinstance(data, list):
            for item in data:
                search_dict(item, target_keys)

    # Start the search
    search_dict(tool_call_output, keys_to_find)

    # Clean up the result: single value if found once, list if multiple, omit if not found
    cleaned_result = {}
    for key, values in result.items():
        if values:  # Only include keys that were found
            cleaned_result[key] = values[0] if len(values) == 1 else values

    return cleaned_result


def replace_placeholder(instruction: str, result: Union[str, bytes]) -> str:
    """
    Replaces the {result} placeholder in an instruction with the actual result.

    Args:
        instruction (str): The instruction containing the {result} placeholder.
        result (Union[str, bytes]): The result to insert into the instruction.

    Returns:
        str: The instruction with the placeholder replaced by the result.
    """
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    return instruction.replace("{result}", result)


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    This is a simple approximation based on word count.
    For more accurate token counting, consider using a tokenizer from the model's library.

    Args:
        text (str): The text to estimate tokens for.

    Returns:
        int: Estimated number of tokens.
    """
    # Simple approximation: ~4 characters per token on average
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within a token limit.

    Args:
        text (str): The text to truncate.
        max_tokens (int): The maximum number of tokens allowed.

    Returns:
        str: The truncated text.
    """
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text

    # Simple approximation: truncate based on character count
    # 4 characters per token is a rough approximation
    max_chars = max_tokens * 4
    truncated = text[:max_chars]

    # Add an indicator that text was truncated
    truncated += "\n\n[Content truncated due to token limit]"

    return truncated


def summarize_for_token_limit(
    text: str, max_tokens: int, model: Optional[Any] = None
) -> str:
    """
    Summarize text to fit within a token limit.

    Args:
        text (str): The text to summarize.
        max_tokens (int): The maximum number of tokens allowed.
        model (Optional[Any]): The model to use for summarization.
            If None, falls back to truncation.

    Returns:
        str: The summarized text.
    """
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text

    if model is None:
        return truncate_to_token_limit(text, max_tokens)

    # Use the provided model to generate a summary
    try:
        # Save the current messages
        original_messages = model.get_messages()

        # Set up summarization prompt
        model.set_messages(
            [
                {
                    "role": "system",
                    "content": f"""You are a summarization assistant. Your task is to create a comprehensive summary 
                of the provided text, preserving all key information, facts, and insights.
                
                The summary should be concise but complete, capturing the essence of the original text.
                Focus on maintaining factual accuracy and including all important details.
                
                Your summary must be shorter than {max_tokens} tokens (approximately {max_tokens * 4} characters).
                """,
                },
                {
                    "role": "user",
                    "content": f"Please summarize the following text:\n\n{text}",
                },
            ]
        )

        # Generate summary
        response = model.generate_response()
        summary = response.get("content", "")

        # Restore original messages
        model.set_messages(original_messages)

        # If summary is still too long, truncate it
        if estimate_tokens(summary) > max_tokens:
            summary = truncate_to_token_limit(summary, max_tokens)

        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        # Fall back to truncation if summarization fails
        return truncate_to_token_limit(text, max_tokens)
