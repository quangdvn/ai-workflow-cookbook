import logging
import os
from datetime import datetime
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s %(levelname)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S'
)


logger = logging.getLogger(__name__)

MODEL = "gpt-4o"


# ============================================================
# Step 1 - Define the data response models for each stage
# ============================================================


class EventExtraction(BaseModel):
  """First LLM call: Extract basic event information"""
  description: str = Field(description="Raw description of the event")
  is_calendar_event: bool = Field(description="Whether this text describes a calendar event")  # Conversation gate
  confidence_score: float = Field(description="Confidence score between 0 and 1")


class EventDetails(BaseModel):
  """Second LLM call: Parse specific event details"""
  name: str = Field(description="Name of the event")
  date: str = Field(
      description="Date and time of the event. Use ISO 8601 to format this value.")
  duration_minutes: int = Field(description="Expected duration in minutes")
  participants: list[str] = Field(description="List of participants")


class EventConfirmation(BaseModel):
  """Third LLM call: Parse specific event details"""
  confirmation_message: str = Field(description="Natural language confirmation message to the user")
  calendar_link: Optional[str] = Field(
    description="Generated link to the calendar event if applicable"
  )

# ============================================================
# Step 2 - Define the functions
# ============================================================


def extract_event_info(user_input: str) -> Optional[EventExtraction]:
  """First LLM call to determine if input is a calendar event"""
  logger.info("Start the event extraction analysis")
  logger.debug(f"Input text: {user_input}")

  today = datetime.now()
  date_context = f"Today is {today.strftime('%A, %B, %d, %Y')}"

  completion = client.beta.chat.completions.parse(
      model=MODEL,
      messages=[
          {
              "role": "system",
              "content": f"{date_context} Analyze if the given describes a calendar event"
          },
          {
              "role": "user",
              "content": user_input
          }
      ],
      response_format=EventExtraction
  )
  result = completion.choices[0].message.parsed

  if result is None:
    logger.error("Failed to parse event extraction")
    return None

  logger.info(
    f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:2f}"
  )
  return result


def parse_event_details(description: str) -> Optional[EventDetails]:
  """Second LLM call to parse event details"""
  logger.info("Start the event details parsing")

  today = datetime.now()
  date_context = f"Today is {today.strftime('%A, %B, %d, %Y')}"

  completion = client.beta.chat.completions.parse(
      model=MODEL,
      messages=[
          {
              "role": "system",
              "content":
              f"""{date_context} Extract detailed event information.
                When dates reference 'next Tuesday' or similar relative dates, use this current date as reference.
              """
          },
          {
              "role": "user",
              "content": description
          }
      ],
      response_format=EventDetails
  )
  result = completion.choices[0].message.parsed

  logger.info(f"Parsing complete - Event details: {result}")
  return result


def generate_confirmation(event_details: EventDetails) -> Optional[EventConfirmation]:
  """Third LLM call to generate a confirmation message"""
  logger.info("Generating the confirmation message")

  completion = client.beta.chat.completions.parse(
      model=MODEL,
      messages=[
          {
              "role": "system",
              "content": "Generate a confirmation message for the event. Sign of with the name, Quang."
          },
          {
              "role": "user",
              "content": str(event_details.model_dump())
          }
      ],
      response_format=EventConfirmation
  )
  result = completion.choices[0].message.parsed

  logger.info(f"Confirmation generation complete - Confirmation: {result}")
  return result

# ============================================================
# Step 3 - Chain the functions together
# ============================================================


def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
  """Main function implements the prompt chain with logic gate check"""
  logger.info("Start the calendar request processing")
  logger.debug(f"Raw input text: {user_input}")

  # First LLM call
  info_extraction = extract_event_info(user_input)

  if info_extraction is None:
    logger.error("Failed to extract event information")
    return None

  # Logic gate check
  if (not info_extraction.is_calendar_event
          or info_extraction.confidence_score < 0.8):
    logger.warning(
      f"Gate check failed - Is calendar event: {info_extraction.is_calendar_event}, Confidence: {info_extraction.confidence_score:2f}"
    )
    return None

  logger.info("Gate check passed, proceeding with further event processing")

  # 2nd LLM call
  event_details = parse_event_details(info_extraction.description)
  if event_details is None:
    logger.error("Failed to parse event details")
    return None

  # 3rd LLM call
  event_confirmation = generate_confirmation(event_details)

  logger.info("Calendar request processing complete")
  return event_confirmation


# ============================================================
# Step 4 - Test the chain with a valid input
# ============================================================
prompt_input = """
Let's schedule a team meeting next Friday, between me in Japan and T from Paris and D from Mumbai to discuss a very important project roadmap.
Since members will join from different time zone, come up with the most suitable time for everyone.
Also, explicit tell everyone at which time they will join from their time zone
"""

final_result = process_calendar_request(prompt_input)
if final_result:
  print(f"Confirmation: {final_result.confirmation_message}")
  if final_result.calendar_link:
    print(f"Calendar Link: {final_result.calendar_link}")
else:
  print("This doesn't appear to be a calendar event request.")


# --------------------------------------------------------------
# Step 5: Test the chain with an invalid input
# --------------------------------------------------------------
prompt_input = "What is Python"
# user_input = "Can you send an email to Alice and Bob to discuss the project roadmap?"

final_result = process_calendar_request(prompt_input)
if final_result is not None:
  print(f"Confirmation: {final_result.confirmation_message}")
  if final_result.calendar_link:
    print(f"Calendar Link: {final_result.calendar_link}")
else:
  print("This doesn't appear to be a calendar event request.")
