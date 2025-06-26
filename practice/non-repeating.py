input = "aabc"

soultion = input[0]

non_repeated_chars = [input[0]]

for _ in input[1:]:

    if _ not in non_repeated_chars:
        non_repeated_chars.append(_)
        soultion += non_repeated_chars[0]

    else:
        non_repeated_chars.remove(_)

    if len(non_repeated_chars) ==0:
        soultion += "#"

print(soultion)
