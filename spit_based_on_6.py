path = "path to yoga test set"
cat_0 = []
cat_1 = []
cat_2 = []
cat_3 = []
cat_4 = []
cat_5 = []
with open(path) as f:
    for line in f:
        x = line.split(",")
        if x[1] == '0':
            if not x[3] in cat_0:
                cat_0.append(x[3])
        if x[1] == '1':
            if not x[3] in cat_1:
                cat_1.append(x[3])
        if x[1] == '2':
            if not x[3] in cat_2:
                cat_2.append(x[3])
        if x[1] == '3':
            if not x[3] in cat_3:
                cat_3.append(x[3])
        if x[1] == '4':
            if not x[3] in cat_4:
                cat_4.append(x[3])
        if x[1] == '5':
            if not x[3] in cat_5:
                cat_5.append(x[3])

print(cat_0)
print(cat_1)
print(cat_2)
print(cat_3)
print(cat_4)
print(cat_5)