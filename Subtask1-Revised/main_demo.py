from unidecode import unidecode
import string
import cPickle as pkl
from copy import deepcopy
import re


__author__ = 'tangy'

'''helpers.py'''

def load_model(name):
    f = open(name,"r")
    obj = pkl.load(f)
    f.close()
    return obj

def perform_or(y_lists):
    num_clfs = len(y_lists)
    num_samples = len(y_lists[0])
    y_pred = [0 for i in xrange(num_samples)]
    for clf_idx in xrange(num_clfs):
        y_pred = [y_pred[i] or y_lists[clf_idx][i] for i in xrange(num_samples)]
    return y_pred

def deconstructor(utter,utter_labels,todel):
    utter_labels = deepcopy(utter_labels)
    utter = deepcopy(utter)
    y_id_new = []
    for i in xrange(len(utter_labels)):
        y_id_temp = []
        for j in xrange(len(utter_labels[i])-1,-1,-1):
            if utter_labels[i][j] == todel:
                del utter[i][j]
                del utter_labels[i][j]
                y_id_temp.insert(0,j)
        y_id_new.append(y_id_temp)
    return utter, y_id_new, utter_labels

def reconstructor(utter_label,y_id_new,toadd):
    utter_label = deepcopy(utter_label)
    for i in xrange(len(y_id_new)):
        for j in xrange(len(y_id_new[i])):
            utter_label[i].insert(y_id_new[i][j],toadd)
    return utter_label

def check_x(word_token,nums,punct,acronyms):
    if not re.match(r'[^@]+@[^@]+\.[^@]+', word_token):
        email = False
    else:
        email = True
    if not re.match(r"(:|;|=)[-pPdD)\\3(/'|\]}>oO]+", word_token):
        smiley = False
    else:
        smiley = True
    return all(word_token[i] in punct for i in xrange(len(word_token))) or all(i in nums or i == '.' for i in word_token) or word_token.startswith('http://') or word_token.startswith('https://') or any(word.lower() == (word_token.lower()) for word in acronyms)  or word_token[0] in ['@','#'] or email or smiley

def run_cleanup(word_token,x_label):
        punct = ['!', '#', '"', '%', '$', "'", '&', ')', '(', '+', '*', '-', ',', '/', '.', ';', ':', '=', '<', '?', '>', '@', '[', ']', '\\', '_', '^', '`', '{', '}', '|', '~']
        numbers = [str(i) for i in xrange(10)]
        # This would work fine if other checks performed first (in order)
        if any(char in punct for char in word_token):
            # Cannot directly replace the following since if they are in isolation then problem
            if x_label == 'NX':
                for punch in punct:
                    word_token = word_token.replace(punch,"")
        return word_token


'''batchio.py'''

def getXs():
    # Creating list of acronyms
    with open('./Resources/AcroNew2.txt','r') as ip:
        acronyms = ip.read().split()
    # Other Disallowed components
    punct = set(string.punctuation)
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return acronyms, punct, nums

class Sklearn_ner_demo:
    """
    Remember :
    input X,y , during training is a list of utterances + list of labels
    input X, during testing is a single utter string in a list - due to the unpacking transform
    output X, during testing is a list of labels, for single utter input, with labels as pointed to by inv_dict
    """


    def __init__(self,mode="test"):
        self.mode = mode
        if mode == "test":
            ner_trained_model = load_model("./Resources/ner_model.pkl")
            self.inv_label_dict = {
                0 : "NNE",
                1 : "NE",
            }
            with open('./Resources/cities.txt','r') as ip:
                self.cities = ip.read().split()
            self.preprocessor_ne = ner_trained_model["preprocessor"]
            self.postprocessor_ne = ner_trained_model["postprocessor"]
            self.clfs = ner_trained_model["clfs"]
            self.preprocessor_ne.trans_pipe[0].clean_type = "test"
            self.preprocessor_ne.trans_pipe[1].extract_type = "test"
    def run(self,X,y=None):
        if self.mode == "test":
            dict_list = []
            if len(X[0]) == 0: # exception
                return []
            for idx in xrange(len(X[0].split())):
                if any(X[0].split()[idx].lower() == ind.lower() for ind in self.cities):
                    dict_list.append(1)
                else:
                    dict_list.append(0)
            X_tf_ne = self.preprocessor_ne.transform(X,y)
            try:
                X_tf_ne = self.postprocessor_ne.transform(X_tf_ne)
            except:
                print X
                print self.preprocessor_ne.transform(X,y)
                exit()
            ypred_lists = [clf.predict(X_tf_ne) for clf in self.clfs]
            y_pred = perform_or(ypred_lists)
            y_pred = perform_or([y_pred,dict_list])
            y_pred_str = [self.inv_label_dict[elem] for elem in y_pred]
            return y_pred_str

class Sklearn_li_demo:
    """
    Remember :
    input X,y , during training is a list of utterances + list of labels
    input X, during testing is a single utter string in a list - due to the unpacking transform
    output X, during testing is a list of labels, for single utter input, with labels as pointed to by inv_dict
    """


    def __init__(self,mode="test"):
        self.mode = mode
        if mode == "test":
            self.inv_label_dict_li = {
                0 : "en",
                1 : "hi",
                2 : "bn",
                3 : "ta",
                4 : "gu",
                5 : "mr",
                6 : "kn",
                7 : "te",
                8 : "ml",
                9 : "MIX",
                10 : "O",
            }

            li_trained_model = load_model("./Resources/li_model.pkl")
            self.preprocessor_li = li_trained_model["preprocessor"]
            self.postprocessor_li = li_trained_model["postprocessor"]
            self.clfs = li_trained_model["clfs"]
            self.preprocessor_li.trans_pipe[0].clean_type = "test"
            self.preprocessor_li.trans_pipe[1].extract_type = "test"


    def run(self,X,y=None):
        if self.mode == "test":
            if len(X[0]) == 0: # exception
                return []
            X_tf_li = self.preprocessor_li.transform(X,y)
            X_tf_li = self.postprocessor_li.transform(X_tf_li)
            ypred_lists = [clf.predict(X_tf_li) for clf in self.clfs]
            y_pred = perform_or(ypred_lists)        #   corrected majority voting - pending
            y_pred_str = [self.inv_label_dict_li[elem] for elem in y_pred]
            return y_pred_str


def run_demo(utter_demo):
    assert utter_demo!=None,"`utter_demo` must not be None"
    ids = [1]
    utters = [utter_demo]
    # Getting Xs
    acronyms,punct,nums = getXs()
    y_ids = []
    X_xnx = []
    y_xnx = []
    # Annotating X
    print "annotating x's"
    for id,utter in zip(ids,utters):
        utter_labels = []
        utter_lst = utter.split()
        for i in xrange(len(utter_lst)):
            # Cleanup any unicode
            utter_lst[i] = str(unidecode(utter_lst[i].decode('utf-8')))
            assert (all(ord(c)<128 for c in utter_lst[i]))
            #   This would work fine if other checks performed first (in order)
            if check_x(utter_lst[i],nums,punct,acronyms) or (re.match('h[aeh]+h\S*',utter_lst[i]) and set(utter_lst[i]) <= xSet):
                utter_labels.append("X")
            else:
                utter_labels.append("NX")
            utter_lst[i] = run_cleanup(utter_lst[i],utter_labels[i])
        X_xnx.append(utter_lst)
        y_ids.append(id)
        y_xnx.append(utter_labels)
    # Deconstruct X's
    print "deconstructing x's"
    X_ne,xnx_bak,_ = deconstructor(X_xnx,y_xnx,"X")
    X_ne_str = [' '.join(x_ne) for x_ne in X_ne]
    # Annotating NE
    print "annotating ne's"
    ner_annotator = Sklearn_ner_demo(mode="test")
    y_ne = [ner_annotator.run([X_ne_str[i]]) for i in xrange(len(X_ne_str))]
    # Deconstruct NE's
    print "deconstructing ne's"
    X_li,ne_bak,_ = deconstructor(X_ne,y_ne,"NE")
    X_li_str = [' '.join(x_li) for x_li in X_li]
    # Annotating Lang
    print "annotating languages"
    li_annotator = Sklearn_li_demo(mode="test")
    y_li =[li_annotator.run([X_li_str[i]]) for i in xrange(len(X_li_str))]
    # Reconstructing NE's
    print "reconstructing ne's"
    y_li_recon_ne = reconstructor(y_li,ne_bak,"NE")
    # Reconstructing X's
    print "reconstructing x's"
    y_li_ne_recon_x = reconstructor(y_li_recon_ne,xnx_bak,"X")
    return ' '.join(y_li_ne_recon_x[0])

print run_demo("I hope ki tumhare paas kafi log honge ! Else you lose Sameer :)")