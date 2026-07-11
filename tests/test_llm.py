import os

from openai import OpenAI


# client = OpenAI(
#             api_key="",
#             base_url="https://integrate.api.nvidia.com/v1",
#         )

# response = client.chat.completions.create(
#             model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
#             messages=[{"role": "user", "content": "Hi, How are you?"}],
#             max_tokens=140,
#         )

# print(response.choices[0].message.content)


client = OpenAI(
            api_key="",
            base_url="https://api.groq.com/openai/v1",
        )

response = client.responses.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            input=[{"role": "user", "content": "Hi, How are you?"}],
            max_output_tokens=140,
        )

print(response.output_text.strip())