# class containing methods for preprocessing the datasets
class pre_processing:
    data = []

    def __init__(self, file_name):

        data = []

        # open input and output files
        with open(file_name) as readIn:

            # iterate over each line in input file
            for line in readIn:
                if (file_name[:16] == "data/winequality"):
                    features = line.split(";")
                else:
                    features = line.split(",")
                data.append(features)

        # dataset-dependent operations
        if (
                file_name == "data/forestfires.csv" or file_name == "data/winequality-red.csv" or file_name == "data/winequality-white.csv"):
            data = self.removeHeaders(data, 1)
        elif (file_name == "data/segmentation.data"):
            data = self.removeHeaders(data, 5)

        # move class to rightmost column
        if (file_name == "data/segmentation.data" or file_name == "data/car.data"):
            data = self.moveColumn(data)

        # remove strings
        data = self.removeStrings(data)
        classification = ["data/segmentation.data",
                          "data/car.data",
                          "data/abalone.data"]

        self.data = data

    # Removes Headers from dataset
    def removeHeaders(self, data, rows):
        for i in range(rows):
            del data[0]

        print("Deleted Header Row")
        return data

    # Moves first column to last column for consistency
    def moveColumn(self, data):
        for i in range(len(data)):
            temp = data[i][0]
            data[i][0] = data[i][-1]
            data[i][-1] = temp
        print("Moved first column to last column")
        return data

    # Removes Strings from dataset
    def removeStrings(self, data):
        stringlist = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                d = data[i][j].strip()

                if (j == len(data[i]) - 1):
                    if (data[i][j].endswith('\n')):
                        data[i][j] = data[i][j][:-1]

                try:
                    data[i][j] = float(d)
                except ValueError:
                    if (d not in stringlist):
                        stringlist.append(d)
                        d = len(stringlist)
                    else:
                        d = stringlist.index(d)
                    data[i][j] = float(d)
        if (len(stringlist) > 0):
            print("Removed Strings")
        return data

    # Converts data into a Value Difference Metric Probabilities for distance calculations
    def processClassification(self, inData):
        # Dictionary for probability conversions
        table = {}
        # Stores all classes for numberical conversions later
        classes = []

        # Generates and maps classes to nested dictinary, sorted by class, attribute column, and individual values
        for c in inData:
            if c[-1] not in classes:
                classes.append(c[-1])
            table.setdefault(classes.index(c[-1]), {})
            for idx, a in enumerate(c[:len(c) - 1]):
                try:
                    table[classes.index(c[-1])][idx + 1][a] += 1
                except:
                    table[classes.index(c[-1])].setdefault(idx + 1, {})
                    table[classes.index(c[-1])][idx + 1].setdefault(a, 1)
                    table[classes.index(c[-1])][idx + 1][a] += 1
        # creates probability table within dictionary
        # print("Classification Probability Table")
        for key in table:
            for x in table[key]:
                total = 0
                for a in table[key][x]:
                    total += table[key][x].get(a)
                for a in table[key][x]:
                    table[key][x][a] /= float(total)
                # print("Class:", key, "Attribute:", x, "Values:", table[key][x])
        # Uses the values in dictionary to convert the input data
        for i, c in enumerate(inData):
            for idx, a in enumerate(c[:len(c) - 1]):
                try:
                    temp = classes.index(c[-1])
                    inData[i][0] = temp
                    inData[i][idx + 1] = table[temp][idx + 1][a]
                except:
                    inData[i][0] = c[-1]
                    inData[i][idx + 1] = table[c[-1]][idx + 1][int(a)]
        return (inData)

    def getData(self):
        return self.data
