# Eligible Elector Project
# Description: A program to check voter eligibility based on age

# Step 1: Ask the User's Age
try:
    age = int(input("How old are you? "))
    
    # Check for invalid age (negative numbers)
    if age < 0:
        print("Oops! Age cannot be negative. Please enter a valid age.")
    else:
        # Step 2: Decide the Eligibility
        if age >= 18:
            print("Congratulations! You are eligible to vote. Go make a difference! 🗳️")
        else:
            years_to_wait = 18 - age
            # Handle singular vs plural for "year"
            if years_to_wait == 1:
                print(f"Oops! You're not eligible yet. But hey, only {years_to_wait} more year to go!")
            else:
                print(f"Oops! You're not eligible yet. But hey, only {years_to_wait} more years to go!")
except ValueError:
    print("Please enter a valid number for your age!")
