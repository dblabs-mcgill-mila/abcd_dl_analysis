import requests 
import json 

var_set =  set()

with open('names.txt', 'r') as f:
    for n in f:
        n = n.strip()
        var_set.add(n)
    
print(f' the number of variables we will search for are {len(var_set)}')

path = 'https://ndar.nih.gov/api/datadictionary/v2/datastructure?source=ABCD%20Release%204.0'

r = requests.get(url = path)

data = r.json()

found_vars = set()
for d in data:
    sn = d['shortName']
    print(sn)
    found_vars.add(sn)
    url = 'https://ndar.nih.gov/api/datadictionary/v2/datastructure/' + str(sn)
    e = requests.get(url)
    ed = e.json()
    for ed_ in ed:
        print(ed_)
        pass
    pass

print(f"total variables found are {len(found_vars)}")