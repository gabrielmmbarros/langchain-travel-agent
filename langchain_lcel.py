import openai
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import os
from dotenv import load_dotenv

# This function loads environment variables from .env file into the application's environment,
# allowing for secure management of sensitive data
load_dotenv()

# Configure the OpenAI API with Azure credentials from environment variables
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_type = os.getenv("AZURE_OPENAI_API_TYPE")


# Define a Pydantic model that will structure the LLM's output
class Destination(BaseModel):
    city = Field("city to visit")
    reason = Field("reason why it is interesting to visit")

# Initializing the language model with LangChain
# This creates a ChatOpenAI instance with specific parameters for generating responses
llm = AzureChatOpenAI(
    azure_deployment = "gpt-4.1-nano",
    openai_api_version = openai.api_version,
    temperature = 0.5
)

# Initialize JSON parser with our data model
parser = JsonOutputParser(pydantic_object=Destination)

# Defining prompt templates for each step in our recommendation chain
city_template = PromptTemplate(
    template="""Suggest a city based on an interest in {interest}.
    {output_format}
    """,
    input_variables=["interest"],
    partial_variables={"output_format": parser.get_format_instructions()},
)

restaurant_template = ChatPromptTemplate.from_template(
    "Suggest popular restaurants among locals in {city}"
)

cultural_template = ChatPromptTemplate.from_template(
    "Suggest cultural activities and places in {city}"
)

# Creating LCEL chains using the | operator for function composition
chain_1 = city_template | llm | parser  # Chain to suggest a city based on interest and parse JSON output
chain_2 = restaurant_template | llm | StrOutputParser()  # Chain to recommend restaurants in the selected city
chain_3 = cultural_template | llm | StrOutputParser()  # Chain to suggest cultural activities in the selected city

# Combining all chains into a sequential pipeline
chain = (chain_1 | chain_2 | chain_3)

# Execute the chain with the user's interest as input
result = chain.invoke({"interest" : "beach"})
print(result)