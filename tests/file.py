from fileinput import filename


with open(filename) as file_object:
    contents = file_object.read()
    print(contents)

with open('pi_digits.txt') as file_object: 
    for line in file_object:
        print(line)

with open('pi_digits.txt') as file_object: 
    lines = file_object.readlines()

pi_string = ''
for line in lines:
    pi_string += line.rstrip()
print(pi_string) 
print(len(pi_string))

filename = 'pi-digits.txt'

with open(filename, 'w') as file_object:
    file_object.write("I love programming.\n")
    file_object.write("I love creating new games.\n")

with open(filename, 'a') as file_object:
    file_object.write("I also love finding meaning in large datasets.\n")
    file_object.write("I love creating apps that can run in a browser.\n")

### ECCEZIONI ###
try:
    print(5/0)
except ZeroDivisionError:
    print("You can't divide by zero!")


print("Give me two numbers, and I'll divide them.")
print("Enter 'q' to quit.")

while True:
    first_number = input("\nFirst number: ")
    if first_number == 'q':
        break
    second_number = input("Second number: ")
    try:
        answer = int(first_number) / int(second_number)
    except ZeroDivisionError: 
        print("You can't divide by 0!")
    else: 
        print(answer)


filename = 'alice.txt'
try:
    with open(filename) as f_obj:
        contents = f_obj.read()
except FileNotFoundError:
    msg = "Sorry, the file " + filename + " does not exist."
    print(msg)
    