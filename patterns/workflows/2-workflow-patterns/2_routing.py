import logging
import os
from datetime import datetime
from typing import Literal, Optional

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
# Step 1: Define the data models for routing and responses
# ============================================================


class CalendarRequestType(BaseModel):
  """Router LLM call: Determine the type of calendar request"""
  request_type: Literal["NEW", "MODIFY", "OTHER"] = Field(
      description="Type of calendar request being made"
  )
  confidence_score: float = Field(description="Confidence score between 0 and 1")
  description: str = Field(description="Cleaned description of the request")


class NewEventDetails(BaseModel):
  """Details for creating a new event"""
  name: str = Field(description="Name of the event")
  date: str = Field(
      description="Date and time of the event. Use ISO 8601 to format this value.")
  duration_minutes: int = Field(description="Expected duration in minutes")
  participants: list[str] = Field(description="List of participants")


class ModifiedDetails(BaseModel):
  """Details for changing an existing event"""
  field: str = Field(description="Field to change")
  new_value: str = Field(description="New value for the field")


class ModifyEventDetails(BaseModel):
  """Details for modifying an existing event"""

  event_identifier: str = Field(
      description="Description to identify the existing event"
  )
  changes: list[ModifiedDetails] = Field(description="List of changes to make")
  participants_to_add: list[str] = Field(description="New participants to add")
  participants_to_remove: list[str] = Field(description="Participants to remove")


class CalendarResponse(BaseModel):
  """Final response format"""
  success: bool = Field(description="Whether the operation was successful")
  message: str = Field(description="User-friendly response message")
  calendar_link: Optional[str] = Field(description="Calendar link if applicable")


# --------------------------------------------------------------
# Step 2: Define the routing and processing functions
# --------------------------------------------------------------

def route_calendar_request(user_input: str) -> Optional[CalendarRequestType]:
  """Router LLM call to determine the type of calendar request"""
  logger.info("Routing calendar request")

  completion = client.beta.chat.completions.parse(
      model=MODEL,
      messages=[
          {
              "role": "system",
              "content": "Determine if this is a request to create a new calendar event or modify an existing one.",
          },
          {"role": "user", "content": user_input},
      ],
      response_format=CalendarRequestType,
  )
  result = completion.choices[0].message.parsed
  if result is None:
    logger.error("Failed to parse event extraction")
    return None

  logger.info(
      f"Request routed as: {result.request_type} with confidence: {result.confidence_score}"
  )
  return result


def handle_new_event(description: str) -> Optional[CalendarResponse]:
  """Process a new event request"""
  logger.info("Processing new event request")

  today = datetime.now()
  date_context = f"Today is {today.strftime('%A, %B, %d, %Y')}"

  # Get event details
  completion = client.beta.chat.completions.parse(
      model=MODEL,
      messages=[
          {
              "role": "system",
              "content":
              f"""{date_context} Extract detailed event information for creating a new calendar event.
                When dates reference 'next Tuesday' or similar relative dates, use this current date as reference.
              """
          },
          {"role": "user", "content": description},
      ],
      response_format=NewEventDetails,
  )
  details = completion.choices[0].message.parsed

  if details is None:
    logger.error("Failed to parse event extraction")
    return None

  logger.info(f"New event: {details.model_dump_json(indent=2)}")

  # Generate response
  return CalendarResponse(
      success=True,
      message=f"Created new event '{details.name}' for {details.date} with {', '.join(details.participants)}",
      calendar_link=f"calendar://new?event={details.name}",
  )


def handle_modify_event(description: str) -> Optional[CalendarResponse]:
  """Process an event modification request"""
  logger.info("Processing event modification request")

  today = datetime.now()
  date_context = f"Today is {today.strftime('%A, %B, %d, %Y')}"

  # Get modification details
  completion = client.beta.chat.completions.parse(
      model=MODEL,
      messages=[
          {
              "role": "system",
              "content":
              f""" {date_context}
								Extract details for modifying an existing calendar event.
								When dates reference 'next Tuesday' or similar relative dates, use this current date as reference.
              """,
          },
          {"role": "user", "content": description},
      ],
      response_format=ModifyEventDetails,
  )
  details = completion.choices[0].message.parsed

  if details is None:
    logger.error("Failed to parse event extraction")
    return None

  logger.info(f"Modified event: {details.model_dump_json(indent=2)}")

  # Generate response
  return CalendarResponse(
      success=True,
      message=f"Modified event '{details.event_identifier}' with the requested changes",
      calendar_link=f"calendar://modify?event={details.event_identifier}",
  )


def process_calendar_request(user_input: str) -> Optional[CalendarResponse]:
  """Main function implementing the routing workflow"""
  logger.info("Processing calendar request")

  # Route the request
  route_result = route_calendar_request(user_input)

  if route_result is None:
    logger.error("Failed to route calendar request")
    return None

  # Check confidence threshold
  if route_result.confidence_score < 0.7:
    logger.warning(f"Low confidence score: {route_result.confidence_score}")
    return None

  # Route to appropriate handler
  if route_result.request_type == "NEW":
    return handle_new_event(route_result.description)
  elif route_result.request_type == "MODIFY":
    return handle_modify_event(route_result.description)
  else:
    logger.warning("Request type not supported")
    return None


# --------------------------------------------------------------
# Step 3: Test with new event
# --------------------------------------------------------------

new_event_input = "Let's schedule a team meeting next Tuesday at 2pm with Alice and Bob"
result = process_calendar_request(new_event_input)
if result:
  print(f"Response: {result.message}")

# --------------------------------------------------------------
# Step 4: Test with modify event
# --------------------------------------------------------------

modify_event_input = (
    "Can you move the team meeting with Alice and Bob to Wednesday at 3pm instead? I think Quang also wants to join as well."
)
result = process_calendar_request(modify_event_input)
if result:
  print(f"Response: {result.message}")

# --------------------------------------------------------------
# Step 5: Test with invalid request
# --------------------------------------------------------------

invalid_input = "A"
result = process_calendar_request(invalid_input)
if not result:
  print("Request not recognized as a calendar operation")
