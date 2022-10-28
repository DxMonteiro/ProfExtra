from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class Summarizer():

    def __init__(self, model):
        self.model = model
        self.summarizer = pipeline("summarization", model=self.model)

    def get_summary(self, text):
        try:
            words = text.split()
            totalwords = len(words)
            summary = self.summarizer(text, max_length=totalwords, do_sample=False)[
                0].get('summary_text')
            return summary
        except:
            print('Error in summarization')
            return 'Error in summarization'
