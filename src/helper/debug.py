# Set initial value to False
FLAG = False

# Function to set the flag
def set_debug():
    global FLAG
    FLAG = True

# Function to get the flag value
def is_debug_on():
    global FLAG
    return FLAG

def change_name(filename: str):
    return "debug_" + filename