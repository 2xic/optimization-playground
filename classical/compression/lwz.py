dictionary_size = 256                   
dictionary = {chr(i): i for i in range(dictionary_size)}    
reverse_dictionary = {}
compressed_data = []
maximum_table_size = 1024

def compress(data):
	global dictionary_size
	string = ""
	for symbol in data:                     
		string_plus_symbol = string + symbol
		if string_plus_symbol in dictionary: 
			string = string_plus_symbol
		else:
			compressed_data.append(dictionary[string])
			if(len(dictionary) <= maximum_table_size):
				dictionary[string_plus_symbol] = dictionary_size
				dictionary_size += 1
			string = symbol
		
	if string in dictionary:
		compressed_data.append(dictionary[string])
	
	for key, value in dictionary.items():
		reverse_dictionary[value] = key
	return compressed_data

def decompress(data):
	for i in data:
		if i in reverse_dictionary:
			print(reverse_dictionary[i])

print(decompress(compress("this is some example text")))
