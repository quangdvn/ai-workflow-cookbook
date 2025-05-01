import os

"""
https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses
"""

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class CalendarEvent(BaseModel):
  name: str
  date: str
  participants: list[str]


completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a calendar assistant, skilled in creating calendar events."},
        {"role": "user", "content": "Create a calendar event for the meeting with John and Jane on 2025-04-22, Tuesday."}
    ],
    response_format=CalendarEvent
)

event = completion.choices[0].message.parsed
event
event.date
event.name
event.participants
