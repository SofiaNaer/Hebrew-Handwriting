import json
occurrences = 9

with open('word_count.json', 'r') as f:
    data = json.load(f)

# Iterate through the data and delete elements with values less than 14
for key in list(data.keys()):
    if data[key] < occurrences:
        del data[key]

with open('word_count_14_and_up.json', 'w') as f:
    json.dump(data, f)
