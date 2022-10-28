from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class Sentimenter():

    def __init__(self, model):
        self.model = model
        self.sentimenter = pipeline("sentiment-analysis", model=self.model)

    def sentiment_analysis(self, text):
        try:
            return self.sentimenter(text)
        except:
            print('Error in sentiment analysis')
            return 'Error in sentiment analysis'
