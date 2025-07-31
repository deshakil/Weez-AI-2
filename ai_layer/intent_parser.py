import os
import json
from openai import AzureOpenAI
from datetime import datetime, timezone

TODAY_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
client = AzureOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    api_version="2024-12-01-preview",
    azure_endpoint="https://weez-openai-resource.openai.azure.com/"
)

def parse_user_intent(raw_query: str) -> dict:
    """
    Parses complex user queries into structured intent including:
    action, platforms, file types, query text, limit, summary type, time range.
    Supports multiple intents per query.
    """

    prompt = f"""
You are a smart assistant that parses user queries for a file AI agent.
Return JSON only. DO NOT include explanations.

Query: "{raw_query}"

Extract the following:
1. action: search / summarize / ask / upload / delete / download / unknown
2. platforms: ["google_drive", "slack", "onedrive", "dropbox", "notion" ...]
3. file_types: ["pdf", "docx", "doc", "txt", "ppt", "pptx", "xls", "xlsx", ...]
4. query_text: the actual content-related intent of the user.
    - This is what the user is trying to find, understand, or describe **about the file**.
    - It must be independent of the action. Do not include "summarize", "search", "rag" etc.
    - Keep it purely about the file’s subject matter, topic, section, or meaning.
    - For example: 
      - ❌ "Summarize the security part" → ✅ "Security-related information"
      - ❌ "Search for Kubernetes issues" → ✅ "Kubernetes scaling or availability issues"
    - Use the user's descriptive language if it reflects the content they care about (not the command).
    - Focus query_text only on the content topic or section — not on the user's command or task.
5. limit: number of files or chunks to consider (default: 5)
6. summary_type: short / general / detailed (default: general)
7. time_range: natural language expression like "last 3 days", "past week", "yesterday", "June 2024"
8. date_range: an object with ISO 8601 format dates like:
   {
     "from": "2025-07-01",
     "to": "2025-07-11"
   }
   - If time_range is present, convert it into exact start & end date
   - Use current date as reference: Today is {TODAY_DATE}
   - If time_range is not present, return an empty string ("") and empty date_range

9. intents: if multiple goals are present, break into separate intents with same structure

Return strictly in this JSON format:
{
  "action": "...",
  "platforms": ["..."],
  "file_types": ["..."],
  "query_text": "...",
  "limit": 5,
  "summary_type": "general",
  "time_range": "last 3 days",
  "date_range": {
    "from": "2025-07-08",
    "to": "2025-07-11"
  },
  "intents": [
    {{
      "action": "...",
      "platforms": ["..."],
      "file_types": ["..."],
      "query_text": "...",
      "limit": 5,
      "summary_type": "short",
      "time_range": "past week",
      "date_range": {
        "from": "2025-07-04",
        "to": "2025-07-11"
      }
    }}
  ]
}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        print("Intent parsing error:", e)
        return {
            "action": "unknown",
            "platforms": [],
            "file_types": [],
            "query_text": raw_query,
            "limit": 5,
            "summary_type": "general",
            "time_range": "",
            "intents": []
        }
