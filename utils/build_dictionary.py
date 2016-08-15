import argparse

parser = argparse.ArgumentParser(description='Build a dictionary')
parser.add_argument('input', metavar='INPUT', type=str, help='path to text')
parser.add_argument('output', metavar='OUTPUT', type=str, help='path to dictionary')
args = parser.parse_args()
dictionary = {}


print("start")
with open(args.input, 'r', encoding='utf8') as input_file:
    for line in input_file:
        for word in line.split():
            if word not in dictionary:
                dictionary[word] = 0
            dictionary[word] += 1

print("dictionary built")
with open(args.output, 'w', encoding='utf8') as output_file:
    output_file.write("\n".join(["{} {}".format(word, freq) for word, freq in dictionary.items()]))

print("done!")
        
