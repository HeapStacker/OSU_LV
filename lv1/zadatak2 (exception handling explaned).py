#ovako funkcionira try catch u pythonu...
# try:
#   print(x)
# except:
#   print("An exception occurred")

print("2. zadatak")
try:
    mark = float(input("Unesi ocjenu: "))
except ValueError:
    print("Value error")
except:
    print("Hell no")
else:
    if mark >= 0.9 and mark <= 1.0:
        print("A")
    elif mark >= 0.8:
        print("B")
    elif mark >= 0.7:
        print("C")
    elif mark >= 0.6:
        print("D")
    elif mark < 0.6:
        print("F")
    else: print("Mark is not in range")

#DALJE SU SAMO PRIMJERI...

#tipovi exceptiona...
# SystemError
# TypeError
# ValueError
# FloatingPointError
# OverflowError
# ZeroDivisionError
# ModuleNotFoundError
# IndexError
# KeyError
# UnboundLocalError
    

#primjer custom exceptiona...
    
# define Python user-defined exceptions
class InvalidAgeException(Exception):
    "Raised when the input value is less than 18"
    pass

#PASS
# The pass statement is used as a placeholder for future code.
# When the pass statement is executed, nothing happens, but you avoid getting an error when empty code is not allowed.
# Empty code is not allowed in loops, function definitions, class definitions, or in if statements.


# you need to guess this number
number = 18

try:
    input_num = int(input("Enter a number: "))
    if input_num < number:
        raise InvalidAgeException
    else:
        print("Eligible to Vote")
        
except InvalidAgeException:
    print("Exception occurred: Invalid Age")




class SalaryNotInRangeError(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, salary, message="Salary is not in (5000, 15000) range"):
        self.salary = salary
        self.message = message
        super().__init__(self.message)


salary = int(input("Enter salary amount: "))
if not 5000 < salary < 15000:
    raise SalaryNotInRangeError(salary)
