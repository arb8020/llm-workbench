import math

def verify_birthday_logic():
    print("=== CORRECT BIRTHDAY LOGIC VERIFICATION ===\n")
    
    # Next birthday is 25/9/2025 (the last square date)
    next_birthday_day = 25
    next_birthday_month = 9
    next_birthday_year = 2025
    
    print(f"Next birthday: {next_birthday_day}/{next_birthday_month}/{next_birthday_year}")
    
    # Current age = sum of square roots
    current_age = math.sqrt(next_birthday_day) + math.sqrt(next_birthday_month) + math.sqrt(next_birthday_year)
    print(f"Current age: √25 + √9 + √2025 = 5 + 3 + 45 = {int(current_age)}")
    
    # If next birthday is in 2025 and current age is 53, then:
    # - Currently it's sometime before the birthday in 2025 (so current year is 2024 or early 2025)
    # - The person will TURN 54 on their birthday in 2025
    # - So they were born in 2025 - 54 = 1971
    
    print(f"\nCORRECT birthday logic:")
    print(f"- Current age: {int(current_age)}")
    print(f"- On next birthday ({next_birthday_day}/{next_birthday_month}/{next_birthday_year}), will turn: {int(current_age) + 1}")
    print(f"- Birth year: {next_birthday_year} - {int(current_age) + 1} = {next_birthday_year - (int(current_age) + 1)}")
    
    birth_year = next_birthday_year - (int(current_age) + 1)
    
    # Verify birth year is in last millennium
    if 1000 <= birth_year <= 1999:
        print(f"✓ Birth year {birth_year} is in last millennium")
    else:
        print(f"✗ Birth year {birth_year} is NOT in last millennium")
        return False
    
    print(f"\n*** CORRECTED FINAL ANSWER ***")
    print(f"I was born in: {birth_year}")
    print(f"My next birthday: {next_birthday_day}/{next_birthday_month}/{next_birthday_year}")
    print(f"My current age: {int(current_age)}")
    print(f"I will turn {int(current_age) + 1} on my next birthday")
    
    # Double check with example
    print(f"\nExample verification:")
    print(f"If someone born in {birth_year} has their birthday on {next_birthday_day}/{next_birthday_month}/{next_birthday_year}:")
    print(f"- They turn {next_birthday_year - birth_year} years old")
    print(f"- Before the birthday, they are {next_birthday_year - birth_year - 1} years old")
    print(f"- Current age {int(current_age)} matches {next_birthday_year - birth_year - 1} ✓")
    
    return birth_year

if __name__ == "__main__":
    birth_year = verify_birthday_logic()
    print(f"\nFINAL ANSWER: Born in {birth_year}")
    print("Mother: 1/8/1936 (unchanged)")