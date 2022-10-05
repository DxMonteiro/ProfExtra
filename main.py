from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from keybert import KeyBERT
import requests
from bs4 import BeautifulSoup
import json

def yake(text):
    resp = requests.get('http://yake.inesctec.pt/yake/v2/extract_keywords?content={}&max_ngram_size=3&number_of_keywords=20&highlight=false'.format(text))
    bsoup = BeautifulSoup(resp.text)
    return json.loads(bsoup.text)

def get_keywords(text):
    kw_model = KeyBERT()
    head = {
        "KeyBERT": kw_model.extract_keywords(text.get('full'), keyphrase_ngram_range=(1, 2), stop_words=None),
        "Yake": yake(text.get('full')).get('keywords')
    }
    return head

def get_summary(text):
    summarizer1 = pipeline("summarization", model="facebook/bart-large-cnn")
    summarizer2 = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summarizer3 = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    summarizer4 = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
    summary = []
    for i in text:
        words = i.split()
        totalwords = len(words)
        head = {
            "facebook/bart-large-cnn": summarizer1(i, max_length = totalwords, do_sample=False)[0].get('summary_text'),
            "sshleifer/distilbart-cnn-12-6": summarizer2(i, max_length = totalwords, do_sample=False)[0].get('summary_text'),
            "philschmid/bart-large-cnn-samsum": summarizer3(i, max_length = totalwords, do_sample=False)[0].get('summary_text'),
            "csebuetnlp/mT5_multilingual_XLSum": summarizer4(i, max_length = totalwords, do_sample=False)[0].get('summary_text'),
        }
        summary.append(head)
    return summary

def process_data(json_data):
    data = []
    for i in json_data:
        head = {
            "title": str(i.get('title')),
            "abstract": i.get('abstract'),
            "content": i.get('content'),
            "keywords": get_keywords(i.get('abstract')),
            "summary": get_summary(i.get('content'))
        }
        data.append(head)
    return data

def write_file(data):
    with open('output.json', 'w') as f:
        json.dump(data, f)

def read_file():
    with open('input.json') as f:
        data = json.load(f)
    return data

def main():
    json_data = read_file()
    data = process_data(json_data)
    write_file(data)

if __name__ == "__main__":
    main()