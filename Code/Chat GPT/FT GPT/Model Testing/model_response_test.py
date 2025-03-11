from openai import OpenAI
from localkey import localkey

OPENAI_API_KEY = localkey

GPT_FT_VERSION = "1"

MODEL_KEYS = {
    "1": "ft:gpt-4o-2024-08-06:personal:ft-gpt-v1:AzPoBReM"
}

USER_PROMPT = {"role": "user", "content": "Evaluate the probability of success for a 19.0 MW Onshore wind farm in Belgium. It is a single phase project beginning in 2019, with a cost of 25821000.0 GBP and is owned by Renner Energies. The project is onshore, and the average annual wind speed is 6.783 m/s. Belgium has a credit rating of 20/21, with a GDP growth of 2.443% and 97.618% of its GDP in debt. The inflation rate is 1.249 and the energy inflation rate is 0.867."}

messages = {
    "1": [{"role": "system", "content": "Classify the following wind farm project into success or failure. First, assess the potential project risks and strengths associated with the wind farm. Then, evaluate both the physical and economic factors and compare these to similar projects. Finally, conclude on if the project is more likey to succeed or fail. Give this response as a single word, 'Success' or 'Fail'."}, USER_PROMPT],
}


client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
    model=MODEL_KEYS[GPT_FT_VERSION],
    messages=messages[GPT_FT_VERSION],
    temperature=0.1,
    top_p=1.0,
)

print(completion.choices)