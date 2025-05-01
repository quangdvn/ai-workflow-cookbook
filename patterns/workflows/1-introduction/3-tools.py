import json
import os

import requests
from openai import OpenAI
from pydantic import BaseModel, Field

"""
https://platform.openai.com/docs/guides/function-calling?api-mode=responses
"""

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_weather(latitude, longitude):
  response = requests.get(
    f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
  data = response.json()
  print("Current: ", data)
  return data['current']


# Step 1 - Call model with get_weather tool defined
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for provided coordinates in celsius.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"}
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

system_prompt = """
        You are a weather assistant, skilled in explaining weather information with exact information in real-time.
        You should give some more useful information about the weather in general, instead of only the exact information asked by user.
"""
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather like in Paris today?"}
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
  if name == "get_weather":
    return get_weather(**args)
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
class WeatherResponse(BaseModel):
  temperature: float = Field(
      description="The current temperature in celsius for the given location."
  )
  response: str = Field(
      description="A natural language response to the user's question."
  )


completion_2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=WeatherResponse
)

# 5 - Check model response
final_response = completion_2.choices[0].message.parsed
final_response.temperature
final_response.response
