import __init__

class Filer():
    
    def __init__(self, inputf):
        self.inputf = inputf

    def write_file(self, data, outputf):
        with open(outputf, 'w') as f:
            json.dump(data, f)

    def read_file(self):
        with open(self.inputf) as f:
            data = json.load(f)
        return data
