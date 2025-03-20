# Assignment: Hands on Python Data Structures

# Task 1 - Working with Lists
print("\n--- Task 1: Working with Lists ---")

# Create a list of favorite fruits
fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry']
print(f"Original list: {fruits}")

# Append a new fruit
fruits.append('fig')
print(f"After adding a fruit: {fruits}")

# Remove a fruit
fruits.remove('apple')  # Removing the first fruit
print(f"After removing a fruit: {fruits}")

# Print the list in reverse order
print(f"Reversed list: {fruits[::-1]}")

# Task 2 - Exploring Dictionaries
print("\n--- Task 2: Exploring Dictionaries ---")

# Create a dictionary with personal information
person = {
    "name": "Alice",
    "age": 25,
    "city": "Boston"
}

# Add a new key-value pair
person["favorite color"] = "Blue"

# Update the city
person["city"] = "New York"

# Print all keys and values
print("Dictionary:", person)
print("Keys:", ", ".join(person.keys()))
print("Values:", ", ".join(str(value) for value in person.values()))

# Task 3 - Using Tuples
print("\n--- Task 3: Using Tuples ---")

# Create a tuple with favorite movie, song, and book
favorites = ('Inception', 'Bohemian Rhapsody', '1984')
print(f"Favorite things: {favorites}")

# Try to change one element (this will cause an error but we'll catch it)
try:
    # This will raise a TypeError
    favorites[0] = 'The Matrix'
except TypeError:
    print("Oops! Tuples cannot be changed.")

# Print the length of the tuple
print(f"Length of tuple: {len(favorites)}")
