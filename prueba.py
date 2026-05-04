import openai

client = openai.OpenAI(
    api_key="s2_de118541a5804ebb880e0a91b19a450c",
    base_url="https://routellm.abacus.ai/v1"
)

modelos = client.models.list()
for m in modelos.data:
    print(m.id)