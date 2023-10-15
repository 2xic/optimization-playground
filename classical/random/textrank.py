"""
https://cran.r-project.org/web/packages/textrank/vignettes/textrank.html

http://web.archive.org/web/20190603083101/https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
"""
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')
stop_words = stopwords.words('english')

class TextRank:
    def __init__(self) -> None:
        self.words_relation = defaultdict(list)
        self.rank_score = defaultdict(int)

    def fit(self, document, n=10):
        self._transform(document)
        self._rank()
        top_words = sorted(self.rank_score.items(), key=lambda x: x[1], reverse=True)
        return top_words[:n]

    def _transform(self, document):
        # Simple normalizing
        words = document.lower().replace("\n", " ").replace("\t", " ").split(" ")
        for index, i in enumerate(words):
            if i not in stop_words:
                if((index + 1) < len(words)):
                    self.words_relation[i] += [words[index + 1]]
                    self.rank_score[i] += 1
                else:
                    self.words_relation[i] = []

    def _rank(self, damp=0.85):
        for word, words_rel in self.words_relation.items():
            new_rank = 0.0
            for word_r in self.words_relation[word]:
                word_relation_count = len(words_rel) 

                if(word_relation_count == 0):
                    word_relation_count = 1
                current_rank = self.rank_score[word_r]
                new_rank += damp * float(current_rank / word_relation_count)
            self.rank_score[word] = new_rank

# Quote from https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
document = """
The Fisher-Yates shuffle is an algorithm for shuffling a finite sequence. The algorithm takes a list of all the elements of the sequence, and continually determines the next element in the shuffled sequence by randomly drawing an element from the list until no elements remain.[1] The algorithm produces an unbiased permutation: every permutation is equally likely. The modern version of the algorithm takes time proportional to the number of items being shuffled and shuffles them in place.

The Fisher-Yates shuffle is named after Ronald Fisher and Frank Yates, who first described it, and is also known as the Knuth shuffle after Donald Knuth.[2] A variant of the Fisher–Yates shuffle, known as Sattolo's algorithm, may be used to generate random cyclic permutations of length n instead of random permutations. 
"""


for key, value in TextRank().fit(document):
    print(f"{key} : {value}")


"""
Output is quite good

fisher-yates : 2.55
next : 1.7
also : 1.7
generate : 1.7
algorithm : 1.0625
finite : 0.85
continually : 0.85
randomly : 0.85
unbiased : 0.85
permutation: : 0.85
"""
