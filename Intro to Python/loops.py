# Assignment: Explore Loops in Python

# Task 1 - Counting Down with Loops
print("\n--- Task 1: Countdown Timer ---")
try:
    start_number = int(input("Enter the starting number: "))
    
    # Check for positive number
    if start_number <= 0:
        print("Please enter a positive number greater than 0.")
    else:
        # Count down using a while loop
        countdown = start_number
        while countdown > 0:
            # Print numbers without a new line
            print(countdown, end=" ")
            countdown -= 1
        print("Blast off! 🚀")
except ValueError:
    print("Please enter a valid number!")

# Task 2 - Multiplication Table with for Loops
print("\n--- Task 2: Multiplication Table ---")
try:
    number = int(input("Enter a number: "))
    
    print(f"Multiplication table for {number}:")
    for i in range(1, 11):
        print(f"{number} x {i} = {number * i}")
except ValueError:
    print("Please enter a valid number!")

# Task 3 - Find the Factorial
print("\n--- Task 3: Factorial Calculator ---")
try:
    factorial_number = int(input("Enter a number: "))
    
    # Check for negative number
    if factorial_number < 0:
        print("Factorial is not defined for negative numbers.")
    else:
        # Calculate factorial using a for loop
        factorial = 1
        for i in range(1, factorial_number + 1):
            factorial *= i
        
        print(f"The factorial of {factorial_number} is {factorial}.")
except ValueError:
    print("Please enter a valid number!")
