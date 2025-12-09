import json

with open("tasks.json") as f:
    tasks = json.load(f)

machines = set()
for task in tasks:
    for op in task['operations']:
        machines.add(op['machine_id'])

print(f"Nombre total de machines: {len(machines)}")
print(f"Liste des machines: {sorted(machines)}")
