import json
import csv

AutomotiveList = []
lines = 0
headerList = ['reviewerID', 'asin', 'overall']
with open('Automotive.json') as f:
    for jsonObj in f:
        AutoDict = json.loads(jsonObj)
        AutomotiveList.append(AutoDict)

with open('Automotive.csv', 'w') as f:
    dw = csv.DictWriter(f, delimiter=',', fieldnames=headerList)
    dw.writeheader()
    for dict in AutomotiveList:
        if dict["verified"] == True:
            f.write("%s,%s,%s\n"%(dict["reviewerID"],dict["asin"],dict["overall"]))
            lines += 1
            # if lines == 500000:
            #     break

print(f"Automotive.csv file is created and has {lines} entries")            