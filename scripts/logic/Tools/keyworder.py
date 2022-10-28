from gensim.summarization import keywords
from keybert import KeyBERT
from rake_nltk import Rake
import yake
import nltk
import nltk.data
import pke

class Keyworder():

    def __init__(self, model):
        self.model = model

    def get_keywords(self, text):
        if self.model == 'KEYBERT':
            result = self.keybert_extractor(text)
        elif self.model == 'TOPIC':
            result = self.topic_rank_extractor(text)
        elif self.model == 'MULTIPARTITE':
            result = self.multipartite_rank_extractor(text)
        elif self.model == 'SINGLE':
            result = self.single_rank_extractor(text)
        elif self.model == 'YAKE':
            result = self.yake_extractor(text)
        elif self.model == 'RAKE':
            result = self.rake_extractor(text)
        elif self.model == 'POSITION':
            result = self.position_rank_extractor(text)
        else:
            result = []

        return result

    # 1. RAKE
    def rake_extractor(self, text):
        """
        Uses Rake to extract the top 5 keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        r = Rake()
        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()[:10]

    # 2. YAKE
    def yake_extractor(self, text):
        """
        Uses YAKE to extract the top 5 keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        keywords = yake.KeywordExtractor(
            lan="en", n=3, windowsSize=3, top=10).extract_keywords(text)
        results = []
        for scored_keywords in keywords:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    # 3. PositionRank
    def position_rank_extractor(self, text):
        """
        Uses PositionRank to extract the top 5 keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        # define the valid Part-of-Speeches to occur in the graph
        pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}
        extractor = pke.unsupervised.PositionRank()
        extractor.load_document(text, language='en')
        extractor.candidate_selection(maximum_word_number=5)
        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk biaised with the position of the words
        #    in the document. In the graph, nodes are words (nouns and
        #    adjectives only) that are connected if they occur in a window of
        #    3 words.
        extractor.candidate_weighting(window=3, pos=pos)
        # 5. get the 5-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)
        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    # 4. SingleRank
    def single_rank_extractor(self, text):
        """
        Uses SingleRank to extract the top 5 keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}
        extractor = pke.unsupervised.SingleRank()
        extractor.load_document(text, language='en')
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(window=3, pos=pos)
        keyphrases = extractor.get_n_best(n=10)
        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    # 5. MultipartiteRank
    def multipartite_rank_extractor(self, text):
        """
        Uses MultipartiteRank to extract the top 5 keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(text, language='en')
        pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(
            alpha=1.1, threshold=0.74, method='average')
        keyphrases = extractor.get_n_best(n=10)
        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    # 6. TopicRank
    def topic_rank_extractor(self, text):
        """
        Uses TopicRank to extract the top 5 keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        extractor = pke.unsupervised.TopicRank()
        extractor.load_document(text, language='en')
        pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting()
        keyphrases = extractor.get_n_best(n=10)
        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    # 7. KeyBERT
    def keybert_extractor(self, text):
        bert = KeyBERT()
        """
        Uses KeyBERT to extract the top 5 keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        keywords = bert.extract_keywords(
            text, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=10)
        results = []
        for scored_keywords in keywords:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results
