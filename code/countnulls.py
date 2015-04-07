import csv
reader = csv.reader(open('mmtd_small_clean.txt'), delimiter='\t')

headers = reader.next()


print headers

head = {}
for h in headers:
    head[h] = 0

for i,line in enumerate(reader):
    for h,col in zip(headers,line):
        if col == 'NULL':
            head[h] += 1

import pprint
for k,v in head.items():
    print k,v

