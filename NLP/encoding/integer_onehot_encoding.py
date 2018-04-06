from numpy import argmax
# define input string
data = 'hello world'
print(data)
# define universe of possible input values
alphabet = 'abcdefghijklmnopqrstuvwxyz '
alphabet_arr = [c for c in alphabet]

# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# integer encode input data
integer_encoded = [char_to_int[char] for char in data]
print("Integer encoding: ")
print(integer_encoded)

# one hot encode
onehot_encoded = list()
for value in integer_encoded:
	letter = [0 for _ in range(len(alphabet))]
	letter[value] = 1
	onehot_encoded.append(letter)

print("One hot encoding: ")
print(alphabet_arr)
for onehot in onehot_encoded:
	print(onehot)

# invert encoding
inverted = int_to_char[argmax(onehot_encoded[0])]
print(inverted)
