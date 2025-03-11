import ollama

DEEPWIND_VERSION = "3a"

USER_PROMPT = {"content": "Evaluate the probability of success for a 40.0 MW Onshore wind farm in Germany. It is a multi phase project beginning in 2015, with a cost of 54360000.0 GBP and is owned by GLS Beteiligung AG. The project is onshore, and the average annual wind speed is 6.757 m/s. Germany has a credit rating of 21/21, with a GDP growth of 1.653% and 70.560% of its GDP in debt. The inflation rate is 0.514 and the energy inflation rate is -0.415.", "role": "user"}
SYSTEM_PROMPT = {"content": "Classify the following wind farm project into success or failure. First, assess the potential project risks and strengths associated with the wind farm. Then, evaluate both the physical and economic factors and compare these to similar projects. Finally, conclude on if the project is more likey to succeed or fail. Give this response as a single word, 'Success' or 'Fail'.", "role": "system"}

message = [SYSTEM_PROMPT, USER_PROMPT]

stream = ollama.chat(
            model=f"deepwind-v{DEEPWIND_VERSION}",
            messages=message,
            stream=True,
        )

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)