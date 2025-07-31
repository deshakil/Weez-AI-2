import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    api_version="2024-12-01-preview",
    azure_endpoint='https://weez-openai-resource.openai.azure.com/'
)

def get_query_embedding(query_text: str) -> list:
    """
    Generates an embedding vector for the user's query_text, 
    which describes the content they are interested in.
    """
    if not query_text:
        raise ValueError("query_text cannot be empty.")

    response = client.embeddings.create(
        input=query_text,
        model='text-embedding-3-large'  # e.g., 'embedding-3-small'
    )

    return response.data[0].embedding
