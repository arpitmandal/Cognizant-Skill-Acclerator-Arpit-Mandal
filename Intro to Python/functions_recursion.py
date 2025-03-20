# Assignment: About Parameters of Functions

# Task 1 - Writing Functions
def greet_user(name):
    """Function to greet a user with their name."""
    return f"Hello, {name}! Welcome aboard."

def add_numbers(a, b):
    """Function to add two numbers and return the result."""
    return a + b

# Test Task 1 functions
print("\n--- Task 1: Writing Functions ---")
user_name = "Alice"
num1, num2 = 5, 10
greeting = greet_user(user_name)
sum_result = add_numbers(num1, num2)
print(f"{greeting} The sum of {num1} and {num2} is {sum_result}.")

# Task 2 - Using Default Parameters
def describe_pet(pet_name, animal_type="dog"):
    """Function to describe a pet with a default animal type."""
    return f"I have a {animal_type} named {pet_name}."

# Test Task 2 function
print("\n--- Task 2: Using Default Parameters ---")
print(describe_pet("Buddy"))  # Using default animal_type
print(describe_pet("Whiskers", "cat"))  # Overriding default parameter

# Task 3 - Functions with Variable Arguments
def make_sandwich(*ingredients):
    """Function to make a sandwich with variable number of ingredients."""
    print("Making a sandwich with the following ingredients:")
    for ingredient in ingredients:
        print(f"- {ingredient}")

# Test Task 3 function
print("\n--- Task 3: Functions with Variable Arguments ---")
make_sandwich("Lettuce", "Tomato", "Cheese")
make_sandwich("Ham", "Swiss Cheese", "Mustard", "Pickles")

# Task 4 - Understanding Recursion
def factorial(n):
    """Calculate the factorial of a number using recursion."""
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def fibonacci(n):
    """Calculate the nth Fibonacci number using recursion."""
    if n <= 0:
        return "Input should be a positive integer"
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Test Task 4 functions
print("\n--- Task 4: Understanding Recursion ---")
fact_num = 5
fib_num = 6
print(f"Factorial of {fact_num} is {factorial(fact_num)}.")
print(f"The {fib_num}th Fibonacci number is {fibonacci(fib_num)}.")
