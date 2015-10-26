import re
import unicodedata
from nltk.util import ngrams
from nltk import pos_tag
from helpers import unpack_list
import numpy as np
from nltk import stem
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from engTextSeparate import *
"""Transform Templates"""

class BaseTransformer:
    """
    Empty Transform
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X,y)

class TransformPipeline(BaseTransformer):
    """
    Pure Transformer Pipeline
    """
    def __init__(self, transform_pipe):
        self.trans_pipe = transform_pipe

    def fit(self,X,y=None):
        for tform in self.trans_pipe:
            tform.fit(X,y)
        return self

    def transform(self, X, y=None):
        Xtf,ytf = X,y
        for tform in self.trans_pipe:
            op = tform.transform(Xtf,ytf)
            if y is not None:
                Xtf,ytf = op
            else:
                Xtf = op
        if y is not None:
            return Xtf,ytf
        else:
            return Xtf

class Model:
    """
    Pure Transformer Pipeline
    """
    def __init__(self, pre_pipe, post_pipe,clf):
        self.pre_pipe = TransformPipeline(pre_pipe)
        self.post_pipe = TransformPipeline(post_pipe)
        self.clf = clf

    def fit(self,X,y):
        X = self.post_pipe.fit_transform(X)
        self.clf.fit(X,y)
        return self

    def predict(self,X):
        X = self.post_pipe.transform(X)
        return self.clf.predict(X)


"""Actual Transforms"""

class Unpacker(BaseTransformer):

    def transform(self, X, y=None):
        X = unpack_list(X)
        if y is not None:
            y = unpack_list(y)
            return X,y
        else:
            return X

class CleanTransform(BaseTransformer):
    """
    3 steps
    1. Removes X's and its corresponding inputs
    2. Removes digits and their corresponding tags
    3. Cleans up other punctuations and unicode characters
    After this transform the data should be processable without errors
    """

    def __init__(self,clean_type,stemm,lem,correct_spellings,label_dict,kick_labels,kick_tokens,failsafe_label):
        self.clean_type = clean_type
        self.stemm=stemm
        self.lem=lem
        self.correct_spellings = correct_spellings
        self.label_dict = label_dict
        self.kick_labels = kick_labels
        self.kick_tokens = kick_tokens
        self.failsafe_label = failsafe_label

    def isAscii(self,utter):
        return all(ord(c)<128 for c in utter)

    def remove_accents(self,data):
        return ''.join(x for x in unicodedata.normalize('NFKD', data) if \
        unicodedata.category(x)[0] == 'L')

    def log_mismatches(self,tmb,tma,bu):
        fp = open("mismatches.log","w")
        fp.write("Token Mismatches - Before : \n")
        for elem in tmb:
            fp.write("%d " % elem)
        fp.write("\n")
        fp.write("Token Mismatches - After : \n")
        for elem in tma:
            fp.write("%d " % elem)
        fp.write("\n")
        fp.write("Blank Utters : \n")
        for elem in bu:
            fp.write("%d " % elem)
        fp.write("\n")
        fp.close()
        print "token mismatches have been logged in `mismatches.log`"

    def printToFile(self,txt):
        with open("Cleaned_data", "a") as myfile:
            np.array(txt).tofile(myfile,",")
            myfile.write("\n")

    def train_clean(self,X,y):
        cleaned_x=[]
        cleaned_y=[]
        num_utter=len(X)
        token_mismatch_before = []
        token_mismatch_after = []
        blank_utterances = []

        for id in xrange(num_utter):
            utter_x=[]
            utter_y=[]
            utter_labels_tokens = y[id].split()
            utter_tokens=X[id].split()
            try:
                assert(len(utter_tokens) == len(utter_labels_tokens))
            except:
                token_mismatch_before.append(id + 1)
                continue
            for i in xrange(len(utter_labels_tokens)):
                if not ((utter_labels_tokens[i] in self.kick_labels) or (utter_tokens[i] in self.kick_tokens) or utter_tokens[i].isdigit()):
                    try:
                        utter_token = utter_tokens[i]
                        if self.stemm:
                            utter_token=stem(utter_token)
                        if self.lem:
                            wordnet_lemmatizer=WordNetLemmatizer()
                            utter_token=wordnet_lemmatizer.lemmatize(utter_token)
                        try:
                            utter_token_label = self.label_dict[utter_labels_tokens[i]]
                        except:
                            utter_token_label = self.failsafe_label
                        unicode_string = utter_token.decode("utf-8")
                        token_corrected = self.remove_accents(unicode_string)
                        utter_x.append(token_corrected)
                        utter_y.append(utter_token_label)
                    except:
                        continue
            sentence_x=(' '.join(utter_x))
            sentence_x=re.sub(r'[@#;,"().*!?:\/\\-]','',sentence_x)
            sentence_x=re.sub(r'[_\']','',sentence_x)
            tokens_x = sentence_x.split()
            tokens_y = utter_y
            if len(utter_x) == 0:
                blank_utterances.append(id+1)
                continue
            if len(tokens_x) != len(tokens_y):
                token_mismatch_after.append(id+1)
                continue
            cleaned_x.append(tokens_x)
            cleaned_y.append(tokens_y)
        print "Unicode errors...corrected\n" \
              "Token Mismatch Errors :...Skipped \n" \
              "%d before;%d after" % (len(token_mismatch_before),len(token_mismatch_after))
        self.log_mismatches(token_mismatch_before,token_mismatch_after,blank_utterances)
        if self.correct_spellings: # may have become buggy
            cleaned_x=[str(' '.join(x)) for x in cleaned_x]
            cleaned=[cleaned_x,cleaned_y]
            ets=engTextSeparate(cleaned)
            ets_cacs=ets.cacs()
            cleaned_x = ets_cacs
        return cleaned_x,cleaned_y  # list of list of tokens/labels

    def test_clean(self,X):
        X_clean = [x.split() for x in X]
        return X_clean

    def transform(self, X, y=None):
        if self.clean_type == "train":
            return self.train_clean(X,y)
        elif self.clean_type == "test":
            return  self.test_clean(X)
        else:
            raise("Incorrect `clean_type` must be either `train` or `test`")

class FeatureExtracter(BaseTransformer):

    def __init__(self,extract_type,context_window=4,unpack=True,return_feat_as_str=True):
        self.unpack = unpack
        self.return_feat_as_str = return_feat_as_str
        self.extract_type = extract_type
        self.context_window = context_window

    def transform(self, X, y=None):
        if self.extract_type == "train":
            return self.train_feature_extractor(X,y)
        elif self.extract_type == "test":
            return self.test_feature_extractor(X)
        else:
            raise("Incorrect `extract_type`")

    def test_feature_extractor(self, X):
        num_samples = len(X)
        Xtf = []
        for idx in xrange(num_samples):
            utter_tokens = X[idx]
            try:
                utter_nerfeats = self.text_to_feat(utter_tokens)
            except:
                print utter_tokens
                exit()
            if self.unpack:
                Xtf += utter_nerfeats
            else:
                Xtf.append(utter_nerfeats)
        return Xtf

    def train_feature_extractor(self, X, y=None):
        num_samples = len(X)
        Xtf = []
        ytf = []
        token_uni_error_list=[]
        uni_error_list=[]

        for idx in xrange(num_samples):
            utter_tokens = X[idx]
            utter_tokens_labels = y[idx]
            try:
                utter_nerfeats = self.text_to_feat(utter_tokens)
            except Exception as e:
                uni_error_list.append((idx+1))
                continue
            try:
                assert len(utter_tokens_labels) == len(utter_nerfeats)
            except AssertionError:
                token_uni_error_list.append((idx+1))
                continue
            if self.unpack:
                Xtf += utter_nerfeats
            else:
                Xtf.append(utter_nerfeats)
            if self.unpack:
                ytf += utter_tokens_labels
            else:
                ytf.append(utter_tokens_labels)
        token_uni_error_list=set(token_uni_error_list)
        uni_error_list=set(uni_error_list)
        print "Unicode Character errors"
        print len((uni_error_list))
        print uni_error_list
        print "Token count + Unicode mismatch errors"
        print len((token_uni_error_list))
        print token_uni_error_list
        print len(token_uni_error_list.difference(uni_error_list))
        return Xtf,ytf

    def text_to_feat(self,sentence_tokens):
        feats = []
        tags=self.postag_feat(sentence_tokens)   # Turn this off and the corresponding ret_pos func call to speed up feature generation
        for idx in xrange(len(sentence_tokens)):
            feat = ["bias"]
            # Current woord
            feat += ["token:" + str(sentence_tokens[idx])]
            feat += ["isupper:" + str(sentence_tokens[idx].isupper())]
            # feat += ["istitle:" + str(sentence_tokens[idx].istitle())]
            feat += ["isdigit:" + str(sentence_tokens[idx].isdigit())]
            # feat += self.location_feat(idx, sentence_tokens)
            feat += ["2gram%d%s" % (i,x) for i,x in enumerate(self.char_ngram(2,sentence_tokens[idx]))]
            feat += ["3gram%d%s" % (i,x) for i,x in enumerate(self.char_ngram(3,sentence_tokens[idx]))]
            feat += ["4gram%d%s" % (i,x) for i,x in enumerate(self.char_ngram(4,sentence_tokens[idx]))]
            feat += ["5gram%d%s" % (i,x) for i,x in enumerate(self.char_ngram(5,sentence_tokens[idx]))]
            feat += ["tags:" + str(self.ret_postag_feats(tags,idx)[0])]
            try:
                feat += ["normalisation:" + str(self.word_norm(sentence_tokens[idx]))]
            except Exception as e:
                raise TypeError('Type unknown')
            # feat += ["compression:" + str(self.word_class_feat(sentence_tokens[idx]))]
            # feat += ["typographic:" + str(self.typographic_feat(sentence_tokens[idx]))]

            # Previous Words
            for itr in xrange(self.context_window):
                if idx > itr:
                    feat += ["prev"+str(idx-itr)+":" + str(sentence_tokens[idx-itr])]
                #     feat += ["previsupper:" + str(sentence_tokens[idx-1].isupper())]
                #     feat += ["previstitle:" + str(sentence_tokens[idx-1].istitle())]
                #     feat += ["previsdigit:" + str(sentence_tokens[idx-1].isdigit())]
                #     # feat += ["prevsuffix3:" + str(sentence_tokens[idx-1][-3:])]
                #     # feat += ["prevsuffix2:" + str(sentence_tokens[idx-1][-2:])]
                #     # feat += ["prevprefix3:" + str(sentence_tokens[idx-1][:3])]
                #     # feat += ["prevprefix2:" + str(sentence_tokens[idx-1][:2])]
                #     feat += map(lambda x:"prev2gram" + x,char_ngram(2,sentence_tokens[idx-1]))
                #     # feat += map(lambda x:"prev3gram" + x,char_ngram(3,sentence_tokens[idx-1]))
                    feat += ["prev"+str(idx-itr)+"tags:" + str(self.ret_postag_feats(tags,(idx-itr))[0])]
                #     try:
                #         feat += ["prevnormalisation" + str(word_norm(sentence_tokens[idx-1]))]
                #     except Exception as e:
                #         raise TypeError('Type unknown')
                #     feat += ["prevcompression:" + str(word_class_feat(sentence_tokens[idx-1]))]
                #     feat += ["prevtypographic:" + str(typographic_feat(sentence_tokens[idx-1]))]
                else:
                    feat += ["BOS_"+str(itr)]

            for itr in xrange(self.context_window,0,-1):
                if idx < len(sentence_tokens) - itr:
                    feat += ["next"+str(itr)+":" + str(sentence_tokens[idx+itr])]
                #     feat += ["nextisupper:" + str(sentence_tokens[idx+1].isupper())]
                #     feat += ["nextistitle:" + str(sentence_tokens[idx+1].istitle())]
                #     feat += ["nextisdigit:" + str(sentence_tokens[idx+1].isdigit())]
                #     # feat += ["prevsuffix3:" + str(sentence_tokens[idx-1][-3:])]
                #     # feat += ["prevsuffix2:" + str(sentence_tokens[idx-1][-2:])]
                #     # feat += ["prevprefix3:" + str(sentence_tokens[idx+1][:3])]
                #     # feat += ["prevprefix2:" + str(sentence_tokens[idx+1][:2])]
                #     feat += map(lambda x:"next2gram" + x,char_ngram(2,sentence_tokens[idx-1]))
                #     # feat += map(lambda x:"next3gram" + x,char_ngram(3,sentence_tokens[idx-1]))
                    feat += ["next"+str(itr)+"tags:" + str(self.ret_postag_feats(tags,(idx+itr))[0])]
                #     try:
                #          feat += ["nextnormalisation:" + str(word_norm(sentence_tokens[idx+1]))]
                #     except Exception as e:
                #          raise TypeError('Type unknown')
                #     feat += ["nextcompression:" + str(word_class_feat(sentence_tokens[idx+1]))]
                #     feat += ["nexttypographic:" + str(typographic_feat(sentence_tokens[idx+1]))]
                else:
                    feat += ["EOS_"+str(itr)]
            if self.return_feat_as_str:
                feat = ' '.join(feat)
            feats.append(feat)
        return feats

    def location_feat(self,idx,sentence_tokens):
        return [str(float(idx)/len(sentence_tokens))]

    def local_context_feat(self,idx,sentence_tokens,num_context_words=2):
        temp_sent_tokens = []
        for i in xrange(num_context_words,0,-1):
            temp_sent_tokens.append('^'*i)
        temp_sent_tokens += sentence_tokens
        for i in xrange(num_context_words):
            temp_sent_tokens.append('$'*(i+1))
        context_tokens = []
        for i in xrange(idx,idx+num_context_words*2+1):
            if i != idx+num_context_words:
                context_tokens.append(temp_sent_tokens[i])
        return context_tokens

    def char_ngram(self,n,word):
        char_tokens = list(word)
        char_ngrams = ngrams(char_tokens,n)  # prefix-suffix is automatically generated here
        return map(lambda x:''.join(x),char_ngrams)

    def word_norm(self,word):
        norm_list=[]
        char_list=list(word)
        for c in char_list:
            if c.isupper():
                norm_list.append('A')
            elif c.islower():
                norm_list.append('a')
            elif c.isdigit():
                norm_list.append('0')
            else:
                raise("Something thats not a char or digit reached here. Check!")
        normalized_word = ''.join(norm_list)
        return [normalized_word]

    def word_class_feat(self,word):
        word=self.word_norm(word)[0]
        word_class=[]
        flag='R'        # arbitrary chosen character
        word_l=list(word)
        for c in word_l:
            if c!=flag:
                word_class.append(c)
                flag=c
        final_class=''.join(word_class)
        return [final_class]

    # Needs fixing
    def typographic_feat(self,word):
        word=self.word_norm(word)[0]
        word_list=list(word)
        if not 'A' in word_list and \
            not '0' in word_list:
            return ["AllSmall"]
        elif not 'a' in word_list and \
            not '0' in word_list:
            return ["AllCaps"]
        elif ('a' in word_list or
            'A' in word_list) and \
            '0' in word_list:
            return ["DigitAlpha"]
        elif word_list[0] == "A":
            return ["InitCap"]
        elif not 'a' in word_list and \
               not 'A' in word_list:
             return ["Onlynumeral"]

    def postag_feat(self,sentence_tokens):
        return pos_tag(sentence_tokens)

    def ret_postag_feats(self,tags,idx):
        return [tags[idx][1]]


