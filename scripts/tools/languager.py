import __init__


class Languager():

    def __init__(self, model):
        self.nlp = spacy.load(model)

    def num_chars(text):
        return len(text)

    def num_words(text):
        my_doc = self.nlp(text)
        token_list = []
        for token in my_doc:
            token_list.append(token.text)
        return len(token_list)

    def unique_words(text):
        my_doc = self.nlp(text)
        out = []
        seen = set()
        for word in my_doc:
            if word.text not in seen:
                out.append(word)
            seen.add(word.text)
        return len(out)

    def points(text):
        
        interrogation = 0
        exclamation = 0
        
        for char in text:
            if char == '!':
                exclamation += 1
            elif char == '?':
                interrogation += 1

        head = {
            'interrogation': interrogation,
            'exclamation': exclamation
        }

        return head

    def word_analysis(text):

        my_doc = self.nlp(text)
        noun = 0
        adjective = 0
        adverb = 0
        verb = 0

        for token in my_doc:
            if token.pos == 'NOUN':
                noun += 1
            elif token.pos == 'ADJ':
                adjective += 1
            elif token.pos == 'ADV':
                adverb += 1
            elif token.pos == 'VERB':
                verb += 1

        lexical_diversity = noun+adjective+adverb+verb
        lexical_density = (lexical_diversirty/num_words(text))*100

        head = {
            '#noun': noun,
            '#adjective': adjective,
            '#adverb': adverb,
            '#verb': verb,
            'lexical_diversity': lexical_diversity,
            'lexical_density': lexical_density
        }

        return head
