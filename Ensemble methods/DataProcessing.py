import csv

def preprocess(file):
    result =[]
    reader = csv.reader(file)
    data = list(reader)

    #deleting the headers
    del data[0]

    for each in data:
        label = each[20]

        if label == '0':
            label = '-1'

        del each[21]
        del each[20]

        each.insert(0, label)
        each = map(int, each)

        result.append(each)
    
    return result
