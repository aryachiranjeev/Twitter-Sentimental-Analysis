import nltk
import pickle
import codecs#to convert utf-8 to latin2
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

nltk.download()

class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers=classifiers
    
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        
        #print(votes)
        return mode(votes)
    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
            
        choice_votes=votes.count(mode(votes))
        print(choice_votes)
        conf=choice_votes/len(votes)
        
        return conf
#short_pos = codecs.open("positive.txt","r", encoding='latin2').read()
#short_neg = codecs.open("negative.txt","r", encoding='latin2').read()
short_pos=open("positive.txt","r").read()
short_neg=open("negative.txt","r").read()

all_words=[]
documents=[]
# j=adjective,r=adverb,v=verb
#allowed_word_types=["J","R","V"]
allowed_word_types=["J"]

for review in short_pos.split("\n"):
    documents.append((review,"pos"))
    words=word_tokenize(review)
    pos=nltk.pos_tag(words)
    for w in pos:
        if(w[1][0] in allowed_word_types:
             all_words.append(w[0].lower())


for review in short_neg.split("\n"):
    documents.append((review,"pos"))
    words=word_tokenize(review)
    pos=nltk.pos_tag(words)
    for w in pos:
        if(w[1][0] in allowed_word_types:#here [1] is word and [0] is pos
             all_words.append(w[0].lower())

save_documents=open("documents.pickle","wb")
pickle.dump(documents,save_documents)
save_documents.close()


"""
for review in short_pos.split('\n'):
    documents.append((review,"pos"))
for review in short_pos.split('\n'):
    documents.append((review,"neg"))

all_words=[]

short_words_pos=word_tokenize(short_pos)
short_words_neg=word_tokenize(short_neg)

for w in short_words_pos:
    all_words.append(w.lower())
    
for w in short_words_neg:
    all_words.append(w.lower())

"""
 
all_words=nltk.FreqDist(all_words)

word_features=list(all_words.keys())[:5000]

save_word_features=open("word_features.pickle","wb")
pickle.dump(word_features,save_word_features)
save_word_features.close()

 
def find_features(document):
    words=word_tokenize(document)
    features={}
    for w in word_features:
        features[w]= (w in words)
    
    return features
    
featuresets=[(find_features(rev),category) for (rev,category) in documents]

random.shuffle(featuresets)

training_set=featuresets[:10000]
test_set=featuresets[10000:]

#NAIVE BAYES 

classifier=nltk.NaiveBayesClassifier.train(training_set)


print("original_accuracy by naive_bayes:",nltk.classify.accuracy(classifier,test_set)*100)

classifier.show_most_informative_features(15)

save_classifier=open("originalnaivebayes.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()

#MNB CLASSIFIER

MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("mnb_accuracy:",nltk.classify.accuracy(MNB_classifier,test_set)*100)

save_classifier=open("MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier,save_classifier)
save_classifier.close()

#USING bernoulliCLASSSIFIER

BN_classifier=SklearnClassifier(BernoulliNB())
BN_classifier.train(training_set)
print("bn_accuracy:",nltk.classify.accuracy(BN_classifier,test_set)*100)

save_classifier=open("BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier,save_classifier)
save_classifier.close()

#LOGISTIC REGRESSION

logistic_regression=SklearnClassifier(LogisticRegression())
logistic_regression.train(training_set)
print("logistic_regression accuracy:",nltk.classify.accuracy(logistic_regression,test_set)*100)

save_classifier=open("LOGISTIC_classifier.pickle","wb")
pickle.dump(LOGISTIC_classifier,save_classifier)
save_classifier.close()

#SGD CLASSIFIER
SGDClassifier=SklearnClassifier(SGDClassifier())
SGDClassifier.train(training_set)
print("SGDClassifier accuracy:",nltk.classify.accuracy(SGDClassifier,test_set)*100)

save_classifier=open("SGD_classifier.pickle","wb")
pickle.dump(SGD_classifier,save_classifier)
save_classifier.close()

#SVC CLASSIFIER
SVC_Classifier=SklearnClassifier(SVC())
SVC_Classifier.train(training_set)
print("SVC_Classifier accuracy:",nltk.classify.accuracy(SVC_Classifier,test_set)*100)

save_classifier=open("SVC_classifier.pickle","wb")
pickle.dump(SVC_classifier,save_classifier)
save_classifier.close()

#LINEARSVC CLASSIFIER

LinearSVC_Classifier=SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)
print("LinearSVC_Classifier accuracy:",nltk.classify.accuracy(LinearSVC_Classifier,test_set)*100)

save_classifier=open("Linear_SVC_classifier.pickle","wb")
pickle.dump(Linear_SVC_classifier,save_classifier)
save_classifier.close()

#NUSVC CLASSIFIER

NuSVC_Classifier=SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)
print("NuSVC_Classifier accuracy:",nltk.classify.accuracy(NuSVC_Classifier,test_set)*100)

save_classifier=open("NuSVC_classifier.pickle","wb")
pickle.dump(NuSVC_classifier,save_classifier)
save_classifier.close()

#VOTED CLASSIFIER

voted_classifier=VoteClassifier(MNB_classifier,BN_classifier,logistic_regression,LinearSVC_Classifier,NuSVC_Classifier)#,SGDClassifier,classifier,SVC_Classifier)

print("voted_classifier accuracy:",nltk.classify.accuracy(voted_classifier,test_set)*100)
print("classfication:",voted_classifier.classify(test_set[0][0]),voted_classifier.confidence(test_set[0][0])*100)
print("classfication:",voted_classifier.classify(test_set[1][0]),voted_classifier.confidence(test_set[1][0])*100)
print("classfication:",voted_classifier.classify(test_set[4][0]),voted_classifier.confidence(test_set[4][0])*100)
print("classfication:",voted_classifier.classify(test_set[3][0]),voted_classifier.confidence(test_set[3][0])*100)

def semtiment(text):
     feats=find_features(text)

     return voted_classifier.classify(feats)
