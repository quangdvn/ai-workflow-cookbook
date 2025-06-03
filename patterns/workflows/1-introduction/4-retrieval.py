import json
import os

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

"""
https://platform.openai.com/docs/guides/function-calling
"""


def search_knowledge(question: str):
  with open("kb.json", "r") as f:
    return json.load(f)


def fallback_answer(question: str):
  return f"Sorry, I can only answer questions related to the e-commerce knowledge base. You asked: {question}"


# Step 1 - Call model with search_knowledge tool defined
tools = [
  {
      "type": "function",
      "function": {
          "name": "search_knowledge",
          "description": "Get the answer to the user's inquiry from the knowledge base.",
          "parameters": {
              "type": "object",
              "properties": {
                  "question": {"type": "string"}
              },
              "required": ["question"],
              "additionalProperties": False
          },
          "strict": True
      }
  },
  {
      "type": "function",
      "function": {
          "name": "fallback_answer",
          "description": "Fallback answer if the question is outside the knowledge base topics.",
          "parameters": {
              "type": "object",
              "properties": {
                  "question": {"type": "string"}
              },
              "required": ["question"],
              "additionalProperties": False
          },
          "strict": True
      }
  }
]

system_prompt = """
                        You are a helpful assistant that answers questions based on the knowledge base for our e-commerce store.
                        Use only the tools provided to answer
                        """
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"}
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

# 2 - Model decides to call function(s)
response.model_dump()


# 3 - Execute get_weather function
def call_function(name, args):
  if name == "search_knowledge":
    return search_knowledge(**args)
  elif name == "fallback_answer":
    return fallback_answer(**args)
  else:
    raise ValueError(f"Unknown function: {name}")


for tool_call in response.choices[0].message.tool_calls:
  name = tool_call.function.name
  args = json.loads(tool_call.function.arguments)
  messages.append(response.choices[0].message)  # Update the conversation - keep the memory

  result = call_function(name, args)
  messages.append({
      "role": "tool",
      "tool_call_id": tool_call.id,
      "content": str(result)
  })


# 4 - Supply result and call model again
class InquiryResponse(BaseModel):
  answer: str = Field(
      description="The answer to user inquiry"
  )
  source: int = Field(
      description="The record id of the answer"
  )


completion_2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=InquiryResponse
)

# 5 - Check model response
final_response = completion_2.choices[0].message.parsed
final_response.answer
final_response.source

# 6 - Question that doesn't trigger the tool
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the current weather in Tokyo?"}
]


completion_3 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools
)

completion_3.model_dump()

for tool_call in completion_3.choices[0].message.tool_calls:
  name = tool_call.function.name
  args = json.loads(tool_call.function.arguments)
  messages.append(completion_3.choices[0].message)
  # Step 1: Execute the fallback function manually
  result = call_function(name, args)

  # Step 2: Add the result back into messages
  messages.append({
      "role": "tool",
      "tool_call_id": tool_call.id,
      "content": str(result)
  })

# Step 3: Call the model again to generate final assistant message
completion_final = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=InquiryResponse  # If you want structured output
)

# Now you can access:
completion_final.choices[0].message.parsed.answer
