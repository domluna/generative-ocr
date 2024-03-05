import base64
import openai
from PIL import Image
import re
from typing import List
from pathlib import Path
from io import BytesIO
import time
import base64
from vertexai.preview.generative_models import GenerativeModel, Part


def load_api_key():
    with open(".env") as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY"):
                return line.split("=")[1].strip()
    return None


SYSTEM_MESSAGE = """You are given an image and a series of question(s). Answer as succinctly as possible. Do not explain your reasoning for the answer, or add any additional wording, just give the answer.

For example, if the question is "What is the date?" and the answer is "January 1, 2022", you would respond with "January 1, 2022" and not "The date is January 1, 2022".

The questions have the following format:

QUESTIONS:
1. ...
2. ...
...

You will answer each question in order. If the answer is a numeric value just return that numeric value. Use the following format for your answer:

ANSWERS:
1. ...
2. ...
...""".strip()


def format_number(string):
    # remove trailing . or ,
    string = string.rstrip(".,").replace(",", "")
    if "." in string:
        return float(string)
    return int(string)


def extract_and_format_numbers(string):
    pattern = r"([\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+)"
    matches = re.findall(pattern, string)
    return [format_number(match) for match in matches]


def parse_response(response: str) -> List[str]:
    """Parses answers from the response string format:

    ANSWERS:
    1. ...
    2. ...
    """
    return re.findall(r"\d+\.\s*(.*)", response)


def ocr_openai(
    path_to_image: Path,
    questions: List[str],
):
    img = Image.open(path_to_image)
    img = img.convert("L")

    width, height = img.size
    print("Image Width: ", width)
    print("Image Height: ", height)

    user_message = "QUESTIONS:\n"
    for i, q in enumerate(questions):
        user_message += f"{i+1}. {q}\n"

    buf = BytesIO()
    img.save(buf, format="JPEG")
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()

    api_key = load_api_key()
    client = openai.OpenAI(api_key=api_key)
    t0 = time.time()
    completion = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message.strip()},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        max_tokens=2048,
        temperature=0.1,
    )
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")
    print(completion)
    content = completion.choices[0].message.content
    total_tokens = completion.usage.total_tokens
    print(f"TOTAL COST: {(total_tokens / 1000) * 0.01}")
    print(
        "Estimated cost based for image tokens on vision pricing calculator from https://openai.com/pricing."
    )
    print("RAW CONTENT")
    print(content)
    print()

    answers = parse_response(content)

    return answers


def ocr_google(
    path_to_image: Path,
    questions: List[str],
):
    img = Image.open(path_to_image)
    width, height = img.size
    scale_factor = 1
    if width < 1000 or height < 1000:
        scale_factor = 2
        img = img.resize((int(width * scale_factor), int(height * scale_factor)))
    print("Image Width: ", width)
    print("Image Height: ", height)
    img = img.convert("L")

    buf = BytesIO()
    img.save(buf, format="JPEG")
    # decode in bytes
    base64_image = base64.b64encode(buf.getvalue())
    model = GenerativeModel("gemini-1.0-pro-vision")
    buf.close()

    user_message = "QUESTIONS:\n"
    for i, q in enumerate(questions):
        user_message += f"{i+1}. {q}\n"

    image = Part.from_data(data=base64.b64decode(base64_image), mime_type="image/jpeg")
    t0 = time.time()
    responses = model.generate_content(
        [SYSTEM_MESSAGE, image, user_message],
    )
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")

    total_cost = 0.0025  # image
    # cost is per character not token
    total_cost += ((len(SYSTEM_MESSAGE) + len(user_message)) / 1000) * 0.000125
    total_cost += (len(responses.text) / 1000) * 0.000375

    print(f"TOTAL COST: {total_cost:.5f}")
    print(
        "Estimated cost based for image tokens on vision pricing calculator from https://cloud.google.com/vertex-ai/pricing."
    )
    print(responses)

    content = responses.text
    print("RAW CONTENT")
    print(content)
    answers = parse_response(content)

    return answers


def ocr(
    path_to_image: Path,
    questions: List[str],
    provider: str = "openai",
):
    if provider == "openai":
        return ocr_openai(path_to_image, questions)
    elif provider == "google":
        return ocr_google(path_to_image, questions)
    else:
        raise ValueError(f"Provider {provider} not supported.")
