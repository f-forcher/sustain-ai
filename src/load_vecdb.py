import weaviate
import os

# Connect to your Weaviate instance
client = weaviate.Client(
    url="http://localhost:8080",
    additional_headers={
        "X-OpenAI-Api-Key": "***REMOVED***"
    }
)

# Check if your instance is live and ready
# This should return `True`
print(client.is_ready())
