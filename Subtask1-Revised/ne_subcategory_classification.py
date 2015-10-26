from batchio import get_data
import wikipedia
from bs4 import BeautifulSoup
import re
import cPickle as pkl
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import os.path
from helpers import run_cleanup_wiki,get_improved_term
from tests import *
from py_bing_search import PyBingSearch

dir_path = "./TestingData"
wiki_result_path = "./Resources/wikiResults"


def ner_sub_category(id_list, a_list, u_list):
    pass_flag = 0       # pass_flag is 1 for "White House" or "Narendra Modi"
    a_list = map(lambda x: x.split(' '), a_list)    # list of lists is obtained
    u_list = map(lambda x: x.split(' '), u_list)
    for utter_idx in range(len(id_list)):
        print "\nWorking on Utter ID: %d" % (utter_idx+1)
        for index in range(len(a_list[utter_idx])):
            if pass_flag:       # if this flag = 1, one index in the current utterance is skipped as two tokens were picked in prev iteration
                pass_flag = 0
                continue
            if a_list[utter_idx][index] == 'NE':
                if index != len(a_list[utter_idx])-1:   # if true, this token isn't last word in the utterance
                    if a_list[utter_idx][index+1] == 'NE':
                        print "\nChecking "+str(u_list[utter_idx][index:index+2])+" (Two tokens in one search) ..."
                        pass_flag = 1
                        clf.two_in_one_flag = 1
                        temp = clf.predict(' '.join(u_list[utter_idx][index:index+2]))
                        if temp == 'RERUN':
                            clf.two_in_one_flag = 0
                            print "Rerunning the tokens individually"
                            print "\nChecking "+str(u_list[utter_idx][index])+"..."
                            a_list[utter_idx][index] = clf.predict(u_list[utter_idx][index])
                            pass_flag = 0
                        else:
                            clf.two_in_one_flag = 0
                            a_list[utter_idx][index] = temp
                            a_list[utter_idx][index+1] = temp
                    else:   # general case: the second token isn't NE. thus individual search
                        print "\nChecking "+u_list[utter_idx][index]+" ..."
                        a_list[utter_idx][index] = clf.predict(u_list[utter_idx][index])
                else: # last word in utterance. So individual search
                    print "\nChecking "+u_list[utter_idx][index]+" ..."
                    a_list[utter_idx][index] = clf.predict(u_list[utter_idx][index])
            else: # token isn't NE
                pass
            clf.found = 0   # resetting this flag which is related to marking of 'en'
    print "Returning final annotation list..."
    return a_list


def prepare_submission_nersub(ids,annotations):
    with open('./TestingData/Ann_NERSub.txt','w') as op:
        # Starting Annotation File here
        op.write("<data>")
        for id,annotation in zip(ids,annotations):
            op.write("\n\t<utterance id=\"%d\">\n" % id)
            op.write("\t\t%s\n" % (' '.join(annotation)))
            op.write("\t</utterance>")
        op.write("\n</data>")
    op.close()


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']#PRP-Pronoun (Person, but it also = PRP)


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] # VBN-Past Participle, VBZ=has


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    # print(tag)
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None


def lemmatizer(token):
    token = token.split(' ')
    ans=[]
    wnl = WordNetLemmatizer()
    posTag=pos_tag(token)
    for i in range(len(posTag)):
        attr = penn_to_wn(posTag[i][1])
        # print(attr)
        if attr:
            # print(wnl.lemmatize(posTag[i][0], attr))
            ans.append(wnl.lemmatize(posTag[i][0], attr))
        else:
            # print(posTag[i][0])
            ans.append(posTag[i][0])
            # print('unchanged')
    ans = map(lambda x: str(x), ans)
    return ' '.join(ans)


def class_verification_files(token, sub_cat):
        if sub_cat == 'NE_P':
            fp = open("%s/NE_P_tokens.txt" % wiki_result_path, 'a')
        elif sub_cat == 'NE_L':
            fp = open("%s/NE_L_tokens.txt" % wiki_result_path, 'a')
        elif sub_cat == 'NE_O':
            fp = open("%s/NE_O_tokens.txt" % wiki_result_path, 'a')
        else:
            fp = open("%s/NE_X_tokens.txt" % wiki_result_path, 'a')
        fp.write(token+'\n')
        fp.close()


class UnsupervisedWikipediaClassifier:

    def __init__(self):
        self.setup_keywords()
        self.confidence_flag = 1
        self.two_in_one_flag = 0
        self.found = 0
        self.do_bing_search_for_err = True
        self.load_wiki_summary()

    def setup_keywords(self):
        self.keywords = {}
        self.keywords["NE_P"]=["god","goddess","sanskrit","feminine","masculine","mean","minister","writer","laureate","his","her","he","she","succeed","precede","age","Education","children","spouse","President","member","family","caste","race","person","priest","teacher","born","gender","male","female","name","academics","Actor","alumnus","Biography","birth","occupation","Character","composer","death","Fellow","footballer","programmer","hacker","guitarist","human","living","musician","painter","Participant","personnel","potter","player","poet","producer","singer","Surname","activist","who","Religion","Awards","Born","Signature","physicist","life"]
        self.keywords["NE_L"]=["place","metropolitan","medieval","province","urban","capital","culture","geography","GDP","independence","nation","subcontinent","route","historic","civilisation","west","southwest","east","northeast","land","ocean","constitutional","democracy","populous","democratic", "parliamentary", "maritime", "border" ,"asteroid","stubs","city","country","geography","Settlement","lake","mountain","municipality","population","Region","Republic","River","State","Suburb","Territory","town","village","Water body", "area"]
        self.keywords["NE_O"]=["newspaper","headquarters","opposition","alliance","government","membership","system","parliament","organisation","Institute", "Advocacy","group","Agency","Business","Club","stub","College","company","Corporation","Legislature","Media","Newspaper","Organization","political","party","software","Team","Union","University","founded","established"]

    def store_wiki_summary(self):
        f = open("./wiki_summary.temp","wb")
        pkl.dump(self.dict, f)
        f.close()

    def load_wiki_summary(self):
        self.dict = {}
        if os.path.isfile("./wiki_summary.temp"):
            f = open("wiki_summary.temp","r")
            self.dict = pkl.load(f)
            f.close()
        else:
            pass

    def extract_wiki_info(self, QUERY):
        try:
            extracted_wiki_info = self.dict[QUERY]
            print "Using cached data.."
        except KeyError:
            print "Specified key was absent in pickle file. Parsing wiki..."
            try:
                barack_obama = wikipedia.WikipediaPage(QUERY)
                title = barack_obama.title
                summary = barack_obama.summary
                categories = ' '.join(barack_obama.categories)

                # section headings
                section_headings = ' '.join(re.findall(r'== ([\s\w]+) ==',barack_obama.content))

                # for infobox
                try:
                    html_content = barack_obama.html()
                    html_soup = BeautifulSoup(html_content,"html.parser")
                    infobox_html = html_soup.find('table',{'class' : 'infobox vcard'})
                    infobox_content = []
                    for tr in infobox_html.find_all('tr'):
                        text_content = tr.text
                        text_content = re.sub('\n',' ',text_content)
                        infobox_content += text_content.split()
                except:
                    infobox_content=[]
                infobox_str = ' '.join(infobox_content)
                extracted_wiki_info=[title.lower(),summary.lower(),categories.lower(),section_headings.lower(),infobox_str.lower()]
                extracted_wiki_info = ' '.join(extracted_wiki_info)
                print "Caching data.."
                self.dict[QUERY] = extracted_wiki_info
            except wikipedia.exceptions.DisambiguationError as exception:
                print "Error: Bing search attempt...."
                improved_term = get_improved_term(QUERY)
                if improved_term == QUERY:
                    print "Bing search did not improve results"
                else:
                    print "Improved term : %s" % improved_term
                    print "Rerunning wikipedia search"
                    wiki_info_bing = self.extract_wiki_info(improved_term)
                    self.dict[QUERY] = wiki_info_bing
                    return wiki_info_bing
                print "Disambiguation Error"
                self.confidence_flag = 0
                if self.two_in_one_flag:
                    print "Got dab error under 2 tokens in 1 search..."
                    return 'T_I_O_F'
                return ' '.join(exception.options[0:6])
            except wikipedia.exceptions.PageError:
                print "Error: Bing search attempt...."
                improved_term = get_improved_term(QUERY)
                if improved_term == QUERY:
                    print "Bing search did not improve results"
                else:
                    print "Improved term : %s" % improved_term
                    print "Rerunning wikipedia search"
                    wiki_info_bing = self.extract_wiki_info(improved_term)
                    self.dict[QUERY] = wiki_info_bing
                    return wiki_info_bing
                print wikipedia.exceptions.PageError
                if self.two_in_one_flag:
                    print "Got page error under 2 tokens in 1 search..."
                    return 'T_I_O_F'
                return 'PNE'        # a random string such that no class can get points and ultimately the token is marked as 'en'

        return extracted_wiki_info

    def predict(self, QUERY):
        first_max = 0
        second_max = 0
        confidence_threshold = 5
        QUERY = run_cleanup_wiki(QUERY)
        wiki_str = self.extract_wiki_info(QUERY)
        if wiki_str == 'T_I_O_F':
            return 'RERUN'
        wiki_str = run_cleanup_wiki(wiki_str)
        wiki_str = wiki_str.encode('ascii', 'ignore')
        # wiki_str = run_cleanup_wiki(wiki_str)
        wiki_str = lemmatizer(wiki_str)
        class_points = {}
        for key in self.keywords:       # initialization of dictionary
            class_points[key] = 0
        for ne_key, keyword_list in self.keywords.iteritems():
            for keyword in keyword_list:
                finding = re.findall(r'\b%ss?\b' % keyword, wiki_str, re.I)
                class_points[ne_key] += len(finding)
                if (len(finding) > 0) or self.found == 1: # true, if atleast one point scored in any class
                    self.found = 1              # check this flag too whether it is reset properly
        print class_points
        if not self.found:
            print "Since all class points are zero, this token is NE_P ..."
            class_verification_files(QUERY, "NE_P")
            return 'en'
        for sub_category, points in class_points.items():
            if points > first_max:
                second_max = first_max
                first_max = points
            elif (points < first_max) and (points >= second_max):
                second_max = points
        diff = first_max - second_max
        if self.confidence_flag:
            print "confidence here"
            if diff >= confidence_threshold:
                final_class = '/'.join([key for key in class_points.keys() if class_points[key] == first_max])
                if len(final_class.split('/'))>1:
                    final_class = final_class.split('/')[0]
            elif self.confidence_flag:
                final_class = 'en'
        else:
            print "no confidence here"
            self.confidence_flag = 1
            final_class = '/'.join([key for key in class_points.keys() if class_points[key] == first_max])
            if len(final_class.split('/'))>1:
                final_class = final_class.split('/')[0]
        class_verification_files(QUERY, final_class)
        print "Final class selected for this token is "+final_class
        return final_class


clf = UnsupervisedWikipediaClassifier()
id_list, annotation_list = get_data("%s/Annotation1.txt" % dir_path)
id_list, utterance_list = get_data("%s/Input1.txt" % dir_path)
final_annotation_list = ner_sub_category(id_list, annotation_list, utterance_list)
print "Storing parsed wiki content into .temp file"
clf.store_wiki_summary()
print "Storing done"
print "Preparing Submission format"
prepare_submission_nersub(id_list, final_annotation_list)
print "Submission format prepared and file saved in specified directory"
print "Running tests"
print run_tests(include_ner_subcat=True)
print annotation_count_test("./TestingData/Ann_NERSub.txt")