import csv


with open('../dataFile/Diabetes/test/Diabetes_test_10.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("\t") for line in stripped if line)
    with open('../dataFile/Diabetes/test/Diabetes_test_10.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('A','B','C','D','E','F','G','H','class'))
        # writer.writerow(('x', 'y', 'z', 'w','class'))
        writer.writerows(lines)
       