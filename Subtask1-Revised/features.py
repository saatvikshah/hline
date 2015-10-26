
# -*- coding: utf-8 -*-

from nltk.util import ngrams
from nltk import pos_tag
__author__ = 'tangy'


import unicodedata

def ner_text_feature(sentence_tokens):
    feats = []
    tags=postag_feat(sentence_tokens)   # Turn this off and the corresponding ret_pos func call to speed up feature generation
    for idx in xrange(len(sentence_tokens)):
        feat = ["bias"]
        # Current woord
        feat += ["token:" + str(sentence_tokens[idx])]
        feat += ["isupper:" + str(sentence_tokens[idx].isupper())]
        feat += ["istitle:" + str(sentence_tokens[idx].istitle())]
        feat += ["isdigit:" + str(sentence_tokens[idx].isdigit())]
        # feat += ["suffix3:" + str(sentence_tokens[idx][-3:])]
        # feat += ["suffix2:" + str(sentence_tokens[idx][-2:])]
        # feat += ["prefix3:" + str(sentence_tokens[idx][:3])]
        # feat += ["prefix2:" + str(sentence_tokens[idx][:2])]
        # feat += location_feat(idx, sentence_tokens)
        feat += ["2gram%d%s" % (i,x) for i,x in enumerate(char_ngram(2,sentence_tokens[idx]))]
        # feat += ["3gram%d%s" % (i,x) for i,x in enumerate(char_ngram(3,sentence_tokens[idx]))]
        # feat += ["4gram%d%s" % (i,x) for i,x in enumerate(char_ngram(4,sentence_tokens[idx]))]
        # feat += ["5gram%d%s" % (i,x) for i,x in enumerate(char_ngram(5,sentence_tokens[idx]))]
        feat += ["tags:" + str(ret_postag_feats(tags,idx)[0])]
        try:
            feat += ["normalisation:" + str(word_norm(sentence_tokens[idx]))]
        except Exception as e:
            raise TypeError('Type unknown')
        # feat += ["compression:" + str(word_class_feat(sentence_tokens[idx]))]
        # feat += ["typographic:" + str(typographic_feat(sentence_tokens[idx]))]

        # Previous Word
        # if idx > 0:
        #     feat += ["prev:" + str(sentence_tokens[idx-1])]
        #     feat += ["previsupper:" + str(sentence_tokens[idx-1].isupper())]
        #     feat += ["previstitle:" + str(sentence_tokens[idx-1].istitle())]
        #     feat += ["previsdigit:" + str(sentence_tokens[idx-1].isdigit())]
        #     # feat += ["prevsuffix3:" + str(sentence_tokens[idx-1][-3:])]
        #     # feat += ["prevsuffix2:" + str(sentence_tokens[idx-1][-2:])]
        #     # feat += ["prevprefix3:" + str(sentence_tokens[idx-1][:3])]
        #     # feat += ["prevprefix2:" + str(sentence_tokens[idx-1][:2])]
        #     feat += map(lambda x:"prev2gram" + x,char_ngram(2,sentence_tokens[idx-1]))
        #     # feat += map(lambda x:"prev3gram" + x,char_ngram(3,sentence_tokens[idx-1]))
        #     feat += ["prevtags:" + str(ret_postag_feats(tags,(idx-1))[0])]
        #     try:
        #         feat += ["prevnormalisation" + str(word_norm(sentence_tokens[idx-1]))]
        #     except Exception as e:
        #         raise TypeError('Type unknown')
        #     feat += ["prevcompression:" + str(word_class_feat(sentence_tokens[idx-1]))]
        #     feat += ["prevtypographic:" + str(typographic_feat(sentence_tokens[idx-1]))]
        # else:
        #     feat += ["BOS"]

        # if idx < len(sentence_tokens) - 1:
        #     feat += ["next:" + str(sentence_tokens[idx+1])]
        #     feat += ["nextisupper:" + str(sentence_tokens[idx+1].isupper())]
        #     feat += ["nextistitle:" + str(sentence_tokens[idx+1].istitle())]
        #     feat += ["nextisdigit:" + str(sentence_tokens[idx+1].isdigit())]
        #     # feat += ["prevsuffix3:" + str(sentence_tokens[idx-1][-3:])]
        #     # feat += ["prevsuffix2:" + str(sentence_tokens[idx-1][-2:])]
        #     # feat += ["prevprefix3:" + str(sentence_tokens[idx+1][:3])]
        #     # feat += ["prevprefix2:" + str(sentence_tokens[idx+1][:2])]
        #     feat += map(lambda x:"next2gram" + x,char_ngram(2,sentence_tokens[idx-1]))
        #     # feat += map(lambda x:"next3gram" + x,char_ngram(3,sentence_tokens[idx-1]))
        #     feat += ["nexttags:" + str(ret_postag_feats(tags,(idx+1))[0])]
        #     try:
        #          feat += ["nextnormalisation:" + str(word_norm(sentence_tokens[idx+1]))]
        #     except Exception as e:
        #          raise TypeError('Type unknown')
        #     feat += ["nextcompression:" + str(word_class_feat(sentence_tokens[idx+1]))]
        #     feat += ["nexttypographic:" + str(typographic_feat(sentence_tokens[idx+1]))]
        # else:
        #     feat += ["EOS"]
        # feat = ' '.join(feat)
        feats.append(feat)
    return feats


def location_feat(idx,sentence_tokens):
    return [str(float(idx)/len(sentence_tokens))]

def local_context_feat(idx,sentence_tokens,num_context_words=2):
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

def char_ngram(n,word):
    char_tokens = list(word)
    char_ngrams = ngrams(char_tokens,n)  # prefix-suffix is automatically generated here
    return map(lambda x:''.join(x),char_ngrams)

def word_norm(word):
    norm_list=[]
    encoding = "utf-8"       # or iso-8859-15, or cp1252, or whatever encoding you use
    byte_string = word         # or simply "caf" before python 3.
    unicode_string = byte_string.decode(encoding)
    word_corrected = remove_accents(unicode_string)
    char_list=list(word_corrected)
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

def word_class_feat(word):
    word=word_norm(word)[0]
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
def typographic_feat(word):
    word=word_norm(word)[0]
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


def remove_accents(data):
    return ''.join(x for x in unicodedata.normalize('NFKD', data) if \
    unicodedata.category(x)[0] == 'L')

def postag_feat(sentence_tokens):
    return pos_tag(sentence_tokens)

def ret_postag_feats(tags,idx):
    return [tags[idx][1]]