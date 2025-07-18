# Chat Model Documents: https://python.langchain.com/v0.3/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.3/docs/integrations/chat/openai/

import os
import getpass


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# Load environment variables from .env
load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

# Create a ChatOpenAI model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.3
)
# Invoke the model with a message
result = llm.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)


