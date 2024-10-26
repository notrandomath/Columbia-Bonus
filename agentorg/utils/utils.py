import tiktoken
import phonenumbers
import Levenshtein

def chunk_string(text, tokenizer, max_length, from_end=True):
    # Initialize the tokenizer
	encoding = tiktoken.get_encoding(tokenizer)
	tokens = encoding.encode(text)
	if from_end:
		chunks = encoding.decode(tokens[-max_length:])
	else:
		chunks = encoding.decode(tokens[:max_length])
	return chunks

def normalize(lst):
		return [float(num)/sum(lst) for num in lst]

def str_similarity(string1, string2):
	try:
		distance = Levenshtein.distance(string1, string2)
		max_length = max(len(string1), len(string2))
		similarity = 1 - (distance / max_length)
	except Exception as err:
		print(err)
		similarity = 0
	return similarity

def possible_email(text):
	possible_domain = ["gmail", "hotmail", "yahoo", "outlook", "icloud", "365", "163", "126"]
	pattern = '[\w\.-]+@'
	if re.search(pattern, text):
		return True
	if any(domain in text for domain in possible_domain):
		return True
	return False

def check_email_validation(email):
	# Make a regular expression
	# for validating an Email
	regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
	# pass the regular expression
	# and the string into the fullmatch() method
	if(re.search(regex, email)):
		print("Valid Email")
		return True
 
	else:
		print("Invalid Email")
		return False
	

def check_phone_validation(phone, language):
	phone_number = ""
	if language == "EN":
		for match in phonenumbers.PhoneNumberMatcher(phone, "US"):
			phone_number = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
	if language == "CN":
		for match in phonenumbers.PhoneNumberMatcher(phone, "CN"):
			phone_number = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
	if phone_number:
		print("Valid Phone Number")
		return True
	else:
		print("Invalid Phone Number")
		return False