__author__ = 'ShubhamTripathi'

import re
from nltk import word_tokenize as wt
from batchio import *
from random import randint

#main = get_training_data()
#cleaned = clean(main['utter_x'], main['utter_y'])

class engTextSeparate:

    cleaned_1=[]
    def __init__(self,cleaned_1):
        self.cleaned_1 = cleaned_1

    def wordSeparateOne(self):
        dict_one = './Resources/eng_dict/emnlp_dict.txt'
        with open(dict_one) as dic1:
            words_one = dic1.readlines()
        pat_t = re.compile(r'\t', re.I|re.M)
        pat_n = re.compile(r'\n', re.I|re.M)
        wrong_words_one =[]
        correct_words_one = []
        for i in words_one:
            split = re.split(pat_t, i)
            split1 = re.split(pat_n,split[1])
            correct_words_one.append(re.sub(r"'",'',split1[0]))
            wrong_words_one.append(split[0])
        return wrong_words_one,correct_words_one
    def wordSeparateTwo(self):
        dict_two = './Resources/eng_dict/Test_Set_3802_Pairs.txt'
        with open(dict_two) as dic2:
            words_two = dic2.readlines()
        pat_correct = re.compile(r'\t.*[|]', re.I|re.M)
        pat_wrong = re.compile(r'[|]',re.I|re.M)
        wrong_words_two =[]
        correct_words_two = []
        for i in words_two:
            split = re.split(pat_correct,i)
            split1 = re.split(pat_wrong,i)
            split_final = re.split(r'\t', split1[0])
            correct_words_two.append(re.sub(r"'",'',split[1].strip()))
            wrong_words_two.append(split_final[1].strip())
        return wrong_words_two,correct_words_two
    def finalWrong(self):
        listOne = self.wordSeparateOne()
        listTwo = self.wordSeparateTwo()
        final_wrong = []
        final_correct = []
        for i in listOne[0]:
            final_wrong.append(i)
        for j in listTwo[0]:
            final_wrong.append(j)
        for i in listOne[1]:
            final_correct.append(i)
        for j in listTwo[1]:
            final_correct.append(j)
        return final_wrong,final_correct
    def finalAppend(self):
        path_input = './Resources/vocabulary_eng/txt_1.txt'
        with open(path_input) as p:
            vocab_new = p.readlines()
        word =[]
        for j in vocab_new:
            char_word = re.split(r'\t',j, re.I|re.M)
            word.append(char_word[0])
        fw = self.finalWrong()
        words = []
        for i in fw[1]:
            words.append(i)
        for j in word:
            words.append(j)
        final_correct = list(set(words))
        final_wrong = fw[0]
        return final_wrong,final_correct
    def totalEngWords(self):
        total_eng = []
        fa = self.finalAppend()
        for i in fa[0]:
            total_eng.append(i)
        for i in fa[1]:
            total_eng.append(i)
        #62,290 number of incorrect and correct english words. If i/p string is one of them(len >2), we classify it as english and give out its correct spelling
        return total_eng

    def cacs(self):
        inputWord =[]
        for i in self.cleaned_1[0]:
            for j in wt(i):
                inputWord.append(j)

        tew = self.totalEngWords()
        engWord = []
        try:
            for i in inputWord:
                if len(i)>2:
                    for j in tew:
                        if i == j:
                            engWord.append(i)
        except:
            print 'Error'
        engWord = list(set(engWord))
        fa = self.finalAppend()
        correct = fa[1]
        fw = self.finalWrong()
        fWrong = fw[0]
        fCorrect = fw[1]
        cleanedData = []
        for i in self.cleaned_1[0]:
            new_sentence = []
            for j in wt(i):
                count = 0
                for k in engWord:
                    if j == k:
                        count +=1
                if count > 0:
                    count_new = 1
                    for l in correct:
                        if j == l:
                            new_sentence.append(j.lower())
                            count_new = 0
                            break
                    if count_new:
                        for l in fWrong:
                            if j == l:
                                new_sentence.append(fCorrect[fWrong.index(l)])
                                break
                else:
                    new_sentence.append(j.lower())
            new_sentence = " ".join(new_sentence)
            cleanedData.append(new_sentence)
        return cleanedData
