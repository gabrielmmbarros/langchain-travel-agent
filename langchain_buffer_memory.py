import openai
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
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

# Initializing the language model with LangChain
# This creates a ChatOpenAI instance with specific parameters for generating responses
llm = AzureChatOpenAI(
    azure_deployment = "gpt-4.1-nano",
    openai_api_version = openai.api_version,
    temperature = 0.2
)


# Initialize a memory buffer to store the last 5 interactions in the conversation
memory = ConversationBufferWindowMemory(k=5)
# Create a conversation chain using the language model and memory buffer
conversation = ConversationChain(llm=llm, verbose=True, memory=memory)


# List of messages for the conversation (translated to English)
messages = [
    "I want to visit a place in Brazil famous for its beaches and culture. Can you recommend one?",
    "What is the best time of year to visit in terms of weather?",
    "What kinds of outdoor activities are available?",
    "Any suggestions for eco-friendly accommodation there?",
    "List 20 other cities with similar features to what we have described so far. Rank them by most interesting, including the one you already suggested.",
    "For the first city you suggested earlier, I want to know 5 restaurants to visit. Answer only with the city name and the restaurant names.",
]

# Run the conversation with each message
for message in messages:
    answer = conversation.predict(input=message)
    print(answer)

print(memory.load_memory_variables({}))