import csv
from urllib import urlopen

card_winrates = {}
with open('Magic Online Results - Theros.csv') as csvfile:
    results_reader = csv.reader(csvfile)
    for row in results_reader:
    	print row[2]
        decklist = urlopen(row[2])
        for line in decklist:
        	print line
            if line == '\n':
                break
            part = line.split(None, 1)
            if part[1].strip() not in card_winrates:
                card_winrates[part[1].strip()] = [0,0]
            card_winrates[part[1].strip()][0] += int(part[0])*int(row[0])
            card_winrates[part[1].strip()][1] += int(part[0])*int(row[1])

f = open('card_winrates.csv', 'w')
for entry in card_winrates:
    f.write('"'+entry + '"'+ ', ' + str(card_winrates[entry][0]) + ', ' +
            str(card_winrates[entry][1]) + '\n')
