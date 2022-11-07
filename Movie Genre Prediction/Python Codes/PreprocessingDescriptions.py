import re

class Preprocessing:
    
    def __init__(self):
        self.desc = []
        self.genre = []
        self.mean = 0
        
    def eliminate_link(self, desc):
        return 1 if not re.match('(.*?)http.*?\s?(.*?)', desc) else 0
    
    def remove_hashtag(self, desc):
        if "#" in desc:
            vocabs = desc.split()
            for i, vocab in enumerate(vocabs):
                if "#" in vocab:
                    vocabs[i] = "HASHTAG"
            return " ".join(vocabs)
        else:
            return desc
        
    def remove_mention(self, desc):
        if "@" in desc:
            vocabs = desc.split()
            for i, vocab in enumerate(vocabs):
                if "@" in vocab:
                    vocabs[i] = "MENTION"
            return " ".join(vocabs)
        else:
            return desc
        
    def remove_punctuation(self, desc):
        return re.sub(r'[^\w\s]','',desc)
    
        
    def preprocess_descriptions(self, desc, genre):
        flag = self.eliminate_link(desc)
        if flag:
            desc = self.remove_hashtag(desc)
            desc = self.remove_mention(desc)
            desc = self.remove_punctuation(desc)
            self.desc.append(desc)
            self.genre.append(genre)

        
        