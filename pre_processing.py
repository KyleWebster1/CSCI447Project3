# class containing methods for preprocessing the datasets
class pre_processing:
    """
        A class used to represent a Radial Basis Function Neural Network

        Attributes
        ----------
        file_name: name of file

        Methods
        -------
        removeHeaders- Removes header from data
        moveColumn- Restructures the data into correct format
        removeStrings- Processes strings into numerical data
        processClassification- Normallizes classification data utilizing probability of value occuring
    """
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
        if (file_name == "data/segmentation.data"):
            data = self.moveColumn(data)

        # remove strings
        #data = self.removeStrings(data)
        classification = ["data/segmentation.data",
                          "data/car.data",
                          "data/abalone.data"]
        if file_name in classification:
            data = self.processClassification(data)
        self.data = data

    def removeHeaders(self, data, rows):
        """
        Removes Headers from dataset
        :param data: input data
        :param rows: rows within the data
        :return: returns processed data
        """
        for i in range(rows):
            del data[0]

        print("Deleted Header Row")
        return data

    def moveColumn(self, data):
        """
        Moves columns into correct layout
        :param data: input data
        :return: processed data
        """
        for i in range(len(data)):
            temp = data[i][0]
            data[i][0] = data[i][-1]
            data[i][-1] = temp
        print("Moved first column to last column")
        return data

    def removeStrings(self, data):
        """
        Processes strings in dataset
        :param data: input data
        :return: processed data
        """
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

    def processClassification(self, inData):
        """
        Normalizes data utilizing a Value Difference Metric of Probabilities
        :param inData: input data
        :return: processed data
        """
        # Dictionary for probability conversions
        table = {}
        # Stores all classes for numberical conversions later
        classes = []

        # Generates and maps classes to nested dictinary, sorted by class, attribute column, and individual values
        for i,c in enumerate(inData):
            if (c[-1].endswith('\n')):
                inData[i][-1] = inData[i][-1][:-1]
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
            temp = classes.index(c[-1])
            for idx, a in enumerate(c[:len(c) - 1]):
                try:
                    inData[i][-1] = temp
                    inData[i][idx] = table[temp][idx + 1][a]
                except:
                    inData[i][-1] = c[-1]
                    inData[i][idx] = table[c[-1]][idx + 1][int(a)]
        return (inData)

    def getData(self):
        return self.data
