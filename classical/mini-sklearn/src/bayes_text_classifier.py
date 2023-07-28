from collections import defaultdict

class BayesTextClassifier:
    def __init__(self) -> None:
        self.features_category = defaultdict(lambda: defaultdict(int))
        self.categories = defaultdict(int)

    def fit(self, documents):
        for (X, category) in documents:
            for word in self._get_features(X):
                self.features_category[word][category] += 1
                self.categories[category] += 1
    
    def predict(self, X):
        probabilities = {}
        for category in list(self.categories.keys()):
            probabilities[category] = self._probability(
                X,
                category
            )
        return probabilities
    
    def _probability(self, features, category):
        category_probability = self.categories[category] / sum(self.categories.values())

        p = 1
        for i in self._get_features(features):
            p *= self._weighted_feature(i, category)
            
        
        return p * category_probability

    def _weighted_feature(self, feature, category):
        category_usage = self.categories[category]
        if category_usage == 0:
            return 0
        feature_usage_in_category = self.features_category[feature][category] / category_usage
        feature_usage_total = sum(self.features_category[feature].values())

        return (
            feature_usage_in_category * feature_usage_total
        )

    def _get_features(self, text):
        return text.lower().split(" ")

if __name__ == "__main__":
    classifier = BayesTextClassifier()
    classifier.fit([
        ["I like trains", "trains"],
        ["Trains", "trains"],
        ["Enough about trains, I want the allspark", "trains"],
        
        ["Boat, I drive the boat", "boat"],
        ["boat is not a train", "boat"],
        ["Enough, I only drive boat", "boat"],
        ["a b c d e f boat !", "boat"]
    ]
    )
    print(classifier.predict("boat"))
    print(classifier.predict("trains"))
