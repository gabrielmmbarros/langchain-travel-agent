import openai
from langchain_openai import AzureChatOpenAI
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


# Initializing the language model with LangChain
# This creates a ChatOpenAI instance with specific parameters for generating responses
llm = AzureChatOpenAI(
    azure_deployment = "gpt-4-1-nano",
    openai_api_version = openai.api_version,
    temperature = 0.5
)

response = llm.invoke(prompt)
print(response.content)