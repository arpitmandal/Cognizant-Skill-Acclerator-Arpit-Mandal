# Assignment: Exploring String Methods

# Task 1 - String Slicing and Indexing
print("\n--- Task 1: String Slicing and Indexing ---")

# Create the string variable
sample_string = "Python is amazing!"

# Extract and print slices
first_word = sample_string[:6]  # First 6 characters
amazing_part = sample_string[10:17]  # The word "amazing"
reversed_string = sample_string[::-1]  # Entire string in reverse

# Print the results
print(f"Original string: {sample_string}")
print(f"First word: {first_word}")
print(f"Amazing part: {amazing_part}")
print(f"Reversed string: {reversed_string}")

# Task 2 - String Methods
print("\n--- Task 2: String Methods ---")

# Create the string
string_with_spaces = " hello, python world! "

# Apply string methods
stripped_string = string_with_spaces.strip()
capitalized_string = stripped_string.capitalize()
replaced_string = stripped_string.replace("world", "universe")
uppercase_string = stripped_string.upper()

# Print the results
print(f"Original string: '{string_with_spaces}'")
print(f"After strip(): '{stripped_string}'")
print(f"After capitalize(): '{capitalized_string}'")
print(f"After replace(): '{replaced_string}'")
print(f"After upper(): '{uppercase_string}'")

# Task 3 - Check for Palindromes
print("\n--- Task 3: Palindrome Checker ---")

# Get input from user
word = input("Enter a word: ").lower()

# Remove spaces if any
word = word.replace(" ", "")

# Check if it's a palindrome using slicing
reversed_word = word[::-1]

if word == reversed_word:
    print(f"Yes, '{word}' is a palindrome!")
else:
    print(f"No, '{word}' is not a palindrome.")
