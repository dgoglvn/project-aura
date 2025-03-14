from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import torch
from typing import Tuple, Optional, List, Dict, Any
import pyfiglet


def load_gemma_model() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """loads the pretrained model and the matching tokenizer from huggingface.co

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: the loaded model and tokenizer
    """
    model_name: str = "google/gemma-3-1b-it"

    try:
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load model (if you have limited GPU memory, consider adding `device_map="auto"`,
        # or use half precision for efficiency -> `torch_dtype=torch.float16`)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        return model, tokenizer
    except Exception as e:
        print(f"an error occurred while trying to load {model_name}: {e}")
        return None, None


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    user_input: str,
) -> str:
    """utilizes gemma 3 model to generate responses bases on the user's input

    Args:
        model (PreTrainedModel): the gemma language model
        tokenizer (PreTrainedTokenizer): the tokenizer for the model
        user_input (str): input from the user
        chat_history (Optional[List[Dict[str, str]]], optional): previous conversation turns as a list of dictionaries. Defaults to None.

    Returns:
        str: gemma 3 model's response
    """
    formatted_prompt: str = ""
    formatted_prompt += f"USER: {user_input}\nASSISTANT: "  # <- the model will generate prediction on its turn

    # tokenize the formatted text
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    attention_mask = inputs["attention_mask"]

    # store the input length to identify where the model's output begins
    input_length = inputs.input_ids.shape[1]

    # generate the response
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )

    # decode the model's response
    response: str = tokenizer.decode(
        outputs[0][input_length:], skip_special_tokens=True
    )

    return response


def run_chatbot() -> Any:
    """loads the model and starts a loop for the conversation. the conversation will
    alternate between the user's input and the model's generated response. loop will
    end when the user types 'exit'

    Returns:
        Any: nothing
    """
    print(pyfiglet.figlet_format("project aura", font="larry3d", width=240))
    print("loading gemma 3 model. this may take a moment...")
    model, tokenizer = load_gemma_model()
    if model is None or tokenizer is None:
        print("exiting due to model loading failure")
        return
    print(
        "gemma 3 model was successfully loaded! type 'exit' to leave the conversation."
    )

    # chat_history: List[Dict[str, str]] = []

    while True:
        user_input = input(">>User: ")
        if user_input.lower() == "exit":
            print("Goodbye 👋")
            break

        response: str = generate_response(model, tokenizer, user_input)
        print(f"GEMMA: {response}")


if __name__ == "__main__":
    run_chatbot()
