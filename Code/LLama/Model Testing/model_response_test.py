import ollama

WINDLLAMA_VERSION = "5b2"

USER_PROMPT = {"content": "Evaluate the probability of success for a 40.0 MW Onshore wind farm in Germany. It is a multi phase project beginning in 2015, with a cost of 54360000.0 GBP and is owned by GLS Beteiligung AG. The project is onshore, and the average annual wind speed is 6.757 m/s. Germany has a credit rating of 21/21, with a GDP growth of 1.653% and 70.560% of its GDP in debt. The inflation rate is 0.514 and the energy inflation rate is -0.415.", "role": "user"}

message = {
    "1": [{"content": "Evaluate the probability of success for the wind farm and classify it as successful or failed.", "role": "system"}, USER_PROMPT],
    "2": [{"content": "Evaluate the probability of success for the wind farm and classify it as successful or failed.", "role": "system"}, USER_PROMPT],
    "3": [{"content": "Classify the wind farm project as successful or failed.", "role": "system"}, USER_PROMPT],
    "4a": [{"content": "Classify the wind farm project as successful or failed.", "role": "system"}, USER_PROMPT],
    "4b": [{"content": "Classify the following wind farm project into success or failure.", "role": "system"}, USER_PROMPT],
    "4c": [{"content": "Classify the following wind farm project into success or failure.", "role": "system"}, USER_PROMPT],
    "5a": [{"content": "Classify the following wind farm project into success or failure. First, assess the potential project risks associated with the wind farm. Then, evaluate both the physical and economic factors and compare these to similar projects. Finally, conclude on if the project will succeed or fail.", "role": "system"}, USER_PROMPT],
    "5b1": [{"content": "Analyse the characteristics of the following wind farm project. Identify relevant technical and economic factors based on historical trends. Then, compare these to similar projects. Finally, determine whether the project characteristics align more closely with successful or unsuccessful projects.", "role": "system"}, USER_PROMPT],
    "5b2": [{"content": "Classify the following wind farm project based on technical and economic indicators. Compare it to historical wind farm data and determine which category (successful or fail) it most closely resembles.", "role": "system"}, USER_PROMPT],
}


stream = ollama.chat(
            model=f"windllama-v{WINDLLAMA_VERSION}",
            messages=message[WINDLLAMA_VERSION],
            stream=True,
        )

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)