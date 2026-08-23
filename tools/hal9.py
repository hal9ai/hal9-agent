from data import DATA
from utils import generate_response

def answer_hal9_questions(user_input):
    response = generate_response(
        [
            {"role": "system", "content": DATA["hal9"]},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
        seed=1,
        reasoning_effort="none",
    )

    return response.choices[0].message.content

answer_hal9_questions_description = {
    "type": "function",
    "function": {
        "name": "answer_hal9_questions",
        "description": "Handles questions related to Hal9 or this chatbot-web capabilities",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string",
                    "description": "Take the user input and pass the same string to the function",
                },
            },
            "required": ["user_input"],
            "additionalProperties": False,
        },
    }
}