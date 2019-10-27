import csv

with open('../iris.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('../iris.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('x', 'y', 'z', 'w', 'class'))
        writer.writerows(lines)