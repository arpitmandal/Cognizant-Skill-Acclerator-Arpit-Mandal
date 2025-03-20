# Project: Password Strength Checker

def check_password_strength(password):
    """
    Check the strength of a password based on multiple criteria.
    Returns a list of issues and a strength score.
    """
    issues = []
    score = 0
    
    # Check 1: Length (at least 8 characters)
    if len(password) < 8:
        issues.append("Your password should be at least 8 characters long.")
    else:
        score += 2
    
    # Check 2: Contains at least one uppercase letter
    if not any(char.isupper() for char in password):
        issues.append("Your password needs at least one uppercase letter.")
    else:
        score += 2
    
    # Check 3: Contains at least one lowercase letter
    if not any(char.islower() for char in password):
        issues.append("Your password needs at least one lowercase letter.")
    else:
        score += 2
    
    # Check 4: Contains at least one digit
    if not any(char.isdigit() for char in password):
        issues.append("Your password needs at least one digit.")
    else:
        score += 2
    
    # Check 5: Contains at least one special character
    special_chars = "!@#$%^&*()-_=+[]{}|;:'\",.<>/?"
    if not any(char in special_chars for char in password):
        issues.append("Your password needs at least one special character.")
    else:
        score += 2
    
    return issues, score

# Main program
print("==== Password Strength Checker ====")
password = input("Enter a password: ")

# Check password strength
issues, score = check_password_strength(password)

# Display results
if not issues:
    print("Your password is strong! 💪")
else:
    print("Password issues:")
    for issue in issues:
        print(f"- {issue}")

# Bonus: Password strength meter
print("\nPassword Strength Score:", end=" ")
if score == 10:
    print("10/10 (Very Strong)")
elif score >= 8:
    print(f"{score}/10 (Strong)")
elif score >= 6:
    print(f"{score}/10 (Moderate)")
elif score >= 4:
    print(f"{score}/10 (Weak)")
else:
    print(f"{score}/10 (Very Weak)")