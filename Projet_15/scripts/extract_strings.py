import sys
import re

def extract_strings(filename, min_len=4):
    with open(filename, "rb") as f:
        data = f.read()
        # Find sequences of printable characters
        strings = re.findall(b"[a-zA-Z0-9_ \.\-\:\,\;\(\)\/]{" + str(min_len).encode() + b",}", data)
        for s in strings:
            try:
                print(s.decode("utf-8"))
            except:
                pass

if __name__ == "__main__":
    extract_strings(sys.argv[1])
