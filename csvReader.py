import csv
print("let's read some CSV files")

def read_cell(y,x):
	with open('correlations.csv', 'r') as f:
		reader = csv.reader(f)
		y_count = 0;
		for n in reader:
			if y_count == y:
				cell = n[x]
				return cell
			y_count += 1

correlations = [];
print("Let's check the correlations:")
for i in range(145):
	correlations.append(read_cell(1,i+2))
#	print(i+1)
#	print(read_cell(0,i+1) + " -> " + read_cell(1,i+1))

unfilteredLen = 145;
correlationPairs = [];
for i in range(145):
	correlationPairs.append([read_cell(0,i+2),read_cell(1,i+2)])

firstTenPairs = correlationPairs[:10]
#print(firstTenPairs)

for _ in range(10):
	for i in range(9):
		a = firstTenPairs[i]
		b = firstTenPairs[i+1]
		if (a[1] > b[1]):
			firstTenPairs[i+1] = a
			firstTenPairs[i] = b
#print(firstTenPairs)

# sort correlations
for _ in range(145):
	for i in range(144):
		a = correlationPairs[i]
		b = correlationPairs[i+1]
		if (a[1] > b[1]):
			correlationPairs[i+1] = a
			correlationPairs[i] = b

for i in range(37):
	print(correlationPairs[i])

#print(correlations[len(correlations) - 1])