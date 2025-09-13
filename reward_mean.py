import json

data = json.load(open('data/tau2/results/final/gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json'))
ff = [(d["task_id"], d["reward_info"]["reward"]) for d in data['simulations'] if d["trial"] == 0]
mean = []

for i in range(50):
    fff = [f[1] for f in ff if f[0] == str(i)]
    mean.append(sum(fff)/len(fff))
    print(i, sum(fff)/len(fff))

print(sum(mean)/len(mean))