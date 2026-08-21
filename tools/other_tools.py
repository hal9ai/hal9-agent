def final_response(final_message):
    print(final_message)
    return final_message

final_response_description = { 
    "type": "function",
    "function": {
        "name": "final_response",
        "description": "Delivers the final user-facing reply. Use immediately for greetings, small talk, and simple questions that need no specialized tools. Also use after other tools have finished collecting what is needed.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "final_message": {
                    "type": "string",
                    "description": "A clear and concise message that directly answers the user (including friendly replies to greetings). Do not mention any tools or internal processes.",
                },
            },
            "required": ["final_message"],
            "additionalProperties": False,
        },
    }
}