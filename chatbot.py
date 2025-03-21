import os
import google.generativeai as genai
from dotenv import load_dotenv

from typing import Any
import speech_recognition as sr
import pyfiglet
import textwrap
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

console = Console()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b", generation_config=generation_config
)

chat_session = model.start_chat(history=[])


def capture_audio() -> Any:
    r = sr.Recognizer()
    mic = sr.Microphone()
    input("Press Enter when you're ready to start speaking...")

    with mic as source:
        print("Recording...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    print("Recording stopped. Processing audio...")
    user_input = r.recognize_google(audio)

    try:
        print(f">>User: {user_input}")
        return user_input
    except sr.UnknownValueError:
        print("Could not understand your audio")
    except sr.RequestError as e:
        print(f"API was unreachable or unresponsive; {e}")


def generate_response(user_input: Any) -> str:
    response = chat_session.send_message(user_input)
    wrapped_text: str = textwrap.fill(response.text)

    return Markdown(wrapped_text)


def run_chatbot() -> Any:
    """loads the model and starts a loop for the conversation. the conversation will
    alternate between the user's input and the model's generated response. loop will
    end when the user types 'exit'

    Returns:
        Any: nothing
    """
    print(pyfiglet.figlet_format("project aura", font="larry3d", width=240))

    while True:
        user_input = capture_audio()

        if user_input.lower() == "exit":
            print("GEMMA: Goodbye 👋")
            break

        print(f"GEMINI: ", end="")
        console.print(generate_response(user_input))


if __name__ == "__main__":
    run_chatbot()
