
filename = 'group.txt'

classes = set()
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        try:
            classes.add(int(line[1]))
        except:
            print(line)
            exit(0)
print(classes)