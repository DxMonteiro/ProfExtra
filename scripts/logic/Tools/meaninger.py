import wikipedia
from PyDictionary import PyDictionary
import nltk
nltk.download('punkt')
import spacy
from spacy.lang.en import English
import en_core_web_sm

class Meaninger():
    
    def __init__(self, model):
        self.dc = PyDictionary()
        self.nlp = spacy.load(model)
        
    def get_meaning_dictionary_py(self, word):
        my_doc = self.nlp(word)
        if len(my_doc) > 1:
            return 'No meaning found'
        return self.dc.meaning(word, disable_errors=True)
    
    def get_meaning_wiki(self, word):
        try: 
            return wikipedia.summary(word, sentences=1)
        except:
            return 'No meaning found'
    
    def get_meaning_dictionary_py(self, word):
        my_doc = self.nlp(word)
        if len(my_doc) > 1:
            return 'No meaning found'
        return self.dc.meaning(word, disable_errors=True)
    
    def get_meaning_summary(self, text, word):
        phrases = nltk.sent_tokenize(text)
        for phrase in phrases:
            if word in phrase:
                return phrase
        return 'No meaning found'
    
    def get_all_meanings(self, text, words):
        data = []
        for word in words:
            head = {
                "keyword": str(word),
                "python_dictionary": self.get_meaning_dictionary_py(str(word)),   
                "wikipedia": self.get_meaning_wiki(str(word)),    
                "context_from_abstract": self.get_meaning_summary(str(text), str(word))
            }
            print(head)
            data.append(head)
        return data