import csv

def returnLabels(filename):
    labels = []
    f = open(filename, 'r')
    reader = csv.reader(f)
    firstLine = True
    for line in reader:
        if firstLine:
            firstLine = False
        else:
            labels.append(line[1])
    return labels




labelsP = returnLabels('outputValidation.csv')
labelsV = returnLabels('valid.csv')

length = len(labelsP)
count = 0
for i in range(length):
    if labelsP[i] == labelsV[i]:
        count += 1

print('Percent Correct:', (count/length))
