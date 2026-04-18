
import sys

def count_braces(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    stack = []
    line_num = 1
    col_num = 1
    for i, char in enumerate(content):
        if char == '\n':
            line_num += 1
            col_num = 1
        else:
            col_num += 1
            
        if char == '{':
            stack.append((line_num, col_num))
        elif char == '}':
            if not stack:
                print(f"Extra closing brace at line {line_num}, col {col_num}")
            else:
                stack.pop()
    
    if stack:
        for line, col in stack:
            print(f"Unclosed open brace starting at line {line}, col {col}")

if __name__ == "__main__":
    count_braces(sys.argv[1])
