import csv
print("let's read some CSV files")

def read_cell(x,y):
	with open('correlations.csv', 'r') as f:
		reader = csv.reader(f)
		y_count = 0;
		for n in reader:
			if y_count == y:
				cell = n[x]
				return cell
			y_count += 1

correlations = [];
print (read_cell(2,2))
