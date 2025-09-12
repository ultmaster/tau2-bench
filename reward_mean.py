import json

data = json.load(open('data/tau2/results/final/gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json'))
ff = [(d["task_id"], d["reward_info"]["reward"]) for d in data['simulations']]
for i in range(50):
    fff = [f[1] for f in ff if f[0] == str(i)]
    print(i, sum(fff)/len(fff))
