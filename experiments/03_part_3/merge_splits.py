import json

part1_path = "./alpaca_eval_70b_part1.json"
part2_path = "./alpaca_eval_70b_part2.json"

part1 = json.load(open(part1_path))
part2 = json.load(open(part2_path))

all_response = part1 + part2
with open("alpaca_eval_70b.json", "w") as f:
    json.dump(all_response, f, indent=2)