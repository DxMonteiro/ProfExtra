import __init__

class Sentimenter():

    def __init__(self, model):
        self.model = model
        self.sentimenter = pipeline("sentiment-analysis", model=self.model)

    def get_summary(self, text):
        sentiment = self.sentimenter(text)
        return sentiment
