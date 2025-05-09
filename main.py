import openai
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# This function loads environment variables from .env file into the application's environment,
# allowing for secure management of sensitive data
load_dotenv()

# Configure the OpenAI API with Azure credentials from environment variables
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_type = os.getenv("AZURE_OPENAI_API_TYPE")

# Variables defining the travel itinerary parameters
number_of_days = 7
number_of_children = 2
activity = "beach"

# Defining the prompt message
prompt = f""" Create a {number_of_days}-day travel itinerary
for a family with {number_of_children} children who enjoy {activity}. """

message_list = [
   {"role": "system", "content": "You are a helpful assistant."},
   {"role": "user", "content": prompt}
]

# Make API call
response = openai.chat.completions.create(
    messages = message_list,
    model = "gpt-4.1-nano"
)

travel_plan = response.choices[0].message.content
print(travel_plan)

# Print token usage information
print(f"Total tokens used: {response.usage.total_tokens}")