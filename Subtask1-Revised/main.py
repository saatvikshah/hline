# -*- coding: utf-8 -*-

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from transforms import *
from helpers import *
import numpy as np
from tests import run_tests,annotation_count_test

extra_training_file_name="extraTraining_Input.txt"
extra_training_annotate_name="extraTraining_Annotation.txt"
NUM_UTTERANCES=2908
NUM_UTTERANCES_EXTRA=1701
xSet = set('hae')
from pprint import pprint


def prepare_submission(ids,annotations):
    with open('./TestingData/Annotation1.txt','w') as op:
        # Starting Annotation File here
        op.write("<data>")
        for id,annotation in zip(ids,annotations):
            op.write("\n\t<utterance id=\"%d\">\n" % id)
            op.write("\t\t%s\n" % (' '.join(annotation)))
            op.write("\t</utterance>")
        op.write("\n</data>")
    op.close()

def cross_validate_sklearn(X,y,target_names,extra_data=None,n=4):
    skf = KFold(len(y),n_folds=n,shuffle=True)
    tfidf = TfidfVectorizer(lowercase=False,use_idf=False,sublinear_tf=True)
    with open('./Resources/cities.txt','r') as ip:
        cities = ip.read().split()
    # For adding training data
    if extra_data is not None:
        X_extra,y_extra = extra_data
        X_extra,y_extra = np.array(X_extra),np.array(y_extra)
    for train_idx,test_idx in skf:
        if len(target_names) == 2:
            clfs = [
                LinearSVC(C=13),
                # SVC(C=11,kernel="linear",probability=True),
                # SGDClassifier(loss="log"),
                LogisticRegression(),
                RandomForestClassifier(n_estimators=10,n_jobs=-1),

            ]
        else:
            clfs = [
                LinearSVC(C=13),
            ]
        print "Setting up"
        X_train,y_train = X[train_idx],y[train_idx]
        if extra_data is not None:
            X_train=np.concatenate((X_train,X_extra),axis=0)
            y_train=np.concatenate((y_train,y_extra),axis=0)

        X_test,y_test = X[test_idx],y[test_idx]
        X_train = tfidf.fit_transform(X_train)
        X_test = tfidf.transform(X_test)
        print "Training.."
        map(lambda x:x.fit(X_train,y_train),clfs)
        print "Testing.."
        ypred_lists = [clf.predict(X_test) for clf in clfs]
        if len(target_names) == 2:
            dict_list = []
            word_list = [str(word.split()[1]).replace('token:','') for word in X.tolist()]
            for word in word_list:
                if any(word.lower() == city.lower() for city in cities):
                    dict_list.append(1)
                else:
                    dict_list.append(0)
            ypred_lists.append(dict_list)
        y_pred = perform_or(ypred_lists)

        print confusion_matrix(y_test,y_pred)
        print(classification_report(y_test,y_pred,target_names=target_names))
        # result = precision_recall_fscore_support(y_test,y_pred)

def run_validation(mode,add_extra_datasets=True):
    ids,X,y = get_training_data("Input1.txt","Annotation1.txt",NUM_UTTERANCES)
    if add_extra_datasets:
        ids_extra,X_extra,y_extra = get_training_data("extraTraining_Input.txt","extraTraining_Annotation.txt",NUM_UTTERANCES_EXTRA)
    if mode == "li":
        label_dict_li = {
            "en" : 0,
            "hi" : 1,
            "bn" : 2,
            "ta" : 3,
            "gu" : 4,
            "mr" : 5,
            "kn" : 6,
            "te" : 7,
            "ml" : 8,
            "MIX" : 9,
            "MIX_kn-en" : 9,
            "MIX_en-ta" : 9,
            "MIX_ta-en" : 9,
            "MIX_hi-en" : 9,
            "MIX_en-hi" : 9,
            "MIX_en-bn" : 9,
            "MIX_bn-en" : 9,
            "MIX_en-kn" : 9,
            "O" : 10
        }
        kick_labels_li = ["X","NE","NE_P","NE_L","NE_O","NE_PA","NE_LA","NE_OA",
                          "NE_X","NE_XA","NE_kn","NE_en","NE_ta","NE_hi","NE_bn",
                          "NE_gu","NE_mr","NE_te","NE_ml"]
        kick_tokens_li = range(100)
        failsafe_label_li = 10
        target_names_li=["en","hi","bn","ta","gu","mr","kn","te","ml","MIX","O"]
        preprocessor_li = TransformPipeline([CleanTransform("train",False,False,False,label_dict_li,kick_labels_li,kick_tokens_li,failsafe_label_li),FeatureExtracter(unpack=True,return_feat_as_str=True,extract_type="train")])
        Xtf_li,ytf_li = preprocessor_li.fit_transform(X,y)
        if add_extra_datasets:
            Xtf_extra,ytf_extra = preprocessor_li.fit_transform(X_extra,y_extra)
            cross_validate_sklearn(np.array(Xtf_li),np.array(ytf_li),target_names_li,(Xtf_extra,ytf_extra),4)
        else:
            cross_validate_sklearn(np.array(Xtf_li),np.array(ytf_li),target_names_li,n=4)
    elif mode == "ne":
        kick_labels_ne = ["X"]
        label_dict_ne = {
            "NE" : 1,
            "NE_P" : 1,
            "NE_L" : 1,
            "NE_O" : 1,
            "NE_PA" : 1,
            "NE_LA" : 1,
            "NE_OA" : 1,
            "NE_X" : 1,
            "NE_XA" : 1,
            "NE_kn" : 1,
            "NE_en" : 1,
            "NE_ta" : 1,
            "NE_hi" : 1,
            "NE_bn" : 1,
            "NE_gu" : 1,
            "NE_mr" : 1,
            "NE_te" : 1,
            "NE_ml" : 1,
        }

        target_names_ne=["non_ne","ne"]
        kick_tokens_ne = map(str,range(1000))
        failsafe_label_ne = 0
        preprocessor_ne = TransformPipeline([CleanTransform("train",False,False,False,label_dict_ne,kick_labels_ne,kick_tokens_ne,failsafe_label_ne),FeatureExtracter(unpack=True,return_feat_as_str=True,extract_type="train")])
        Xtf_ne,ytf_ne = preprocessor_ne.fit_transform(X,y)
        if add_extra_datasets:
            Xtf_extra,ytf_extra = preprocessor_ne.fit_transform(X_extra,y_extra)
            cross_validate_sklearn(np.array(Xtf_ne),np.array(ytf_ne),target_names_ne,(Xtf_extra,ytf_extra),4)
        else:
            cross_validate_sklearn(np.array(Xtf_ne),np.array(ytf_ne),target_names_ne,n=4)

class Sklearn_ner:
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
        elif mode == "train":
            kick_labels_ne = ["X"]
            label_dict_ne = {
                "NE" : 1,
                "NE_P" : 1,
                "NE_L" : 1,
                "NE_O" : 1,
                "NE_PA" : 1,
                "NE_LA" : 1,
                "NE_OA" : 1,
                "NE_X" : 1,
                "NE_XA" : 1,
                "NE_kn" : 1,
                "NE_en" : 1,
                "NE_ta" : 1,
                "NE_hi" : 1,
                "NE_bn" : 1,
                "NE_gu" : 1,
                "NE_mr" : 1,
                "NE_te" : 1,
                "NE_ml" : 1,
            }
            kick_tokens_ne = map(str,range(1000))
            failsafe_label_ne = 0
            self.preprocessor_ne = TransformPipeline([CleanTransform("train",False,False,False,label_dict_ne,kick_labels_ne,kick_tokens_ne,failsafe_label_ne),FeatureExtracter(unpack=True,return_feat_as_str=True,extract_type="train")])
            self.postprocessor_ne = TfidfVectorizer(lowercase=False,use_idf=False,sublinear_tf=True)
            self.clfs = [
                LinearSVC(C=13),
                # SVC(C=11,kernel="linear",probability=True),
                # SGDClassifier(loss="log"),
                LogisticRegression(),
                RandomForestClassifier(n_estimators=10,n_jobs=-1),
            ]

    def run(self,X,y=None):
        if self.mode == "train":
            Xtf_ne,ytf_ne = self.preprocessor_ne.fit_transform(X,y)
            Xtf_ne = self.postprocessor_ne.fit_transform(Xtf_ne)
            print "Training.."
            map(lambda x:x.fit(Xtf_ne,ytf_ne),self.clfs)
            ner_trained_model = {
                "preprocessor" : self.preprocessor_ne,
                "postprocessor" : self.postprocessor_ne,
                "clfs" : self.clfs,
            }
            dump_model(ner_trained_model,"./Resources/ner_model.pkl")
            print "Dumped NER model at `./Resources/ner_model.pkl``"
        elif self.mode == "test":
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

class Sklearn_li:
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
        elif mode == "train":
            label_dict_li = {
                "en" : 0,
                "hi" : 1,
                "bn" : 2,
                "ta" : 3,
                "gu" : 4,
                "mr" : 5,
                "kn" : 6,
                "te" : 7,
                "ml" : 8,
                "MIX" : 9,
                "MIX_kn-en" : 9,
                "MIX_en-ta" : 9,
                "MIX_ta-en" : 9,
                "MIX_hi-en" : 9,
                "MIX_en-hi" : 9,
                "MIX_en-bn" : 9,
                "MIX_bn-en" : 9,
                "MIX_en-kn" : 9,
                "O" : 10
            }
            kick_labels_li = ["X","NE","NE_P","NE_L","NE_O","NE_PA","NE_LA","NE_OA",
                              "NE_X","NE_XA","NE_kn","NE_en","NE_ta","NE_hi","NE_bn",
                              "NE_gu","NE_mr","NE_te","NE_ml"]
            kick_tokens_li = range(100)
            failsafe_label_li = 9
            self.preprocessor_li = TransformPipeline([CleanTransform("train",False,False,False,label_dict_li,kick_labels_li,kick_tokens_li,failsafe_label_li),FeatureExtracter(unpack=True,return_feat_as_str=True,extract_type="train")])
            self.postprocessor_li = TfidfVectorizer(lowercase=False,use_idf=False,sublinear_tf=True)
            self.clfs = [
                LinearSVC(C=13),
                # SVC(C=11,kernel="linear",probability=True),
                # SGDClassifier(loss="log"),
                # LogisticRegression(),
                # RandomForestClassifier(n_estimators=10,n_jobs=-1),
            ]


    def run(self,X,y=None):
        if self.mode == "train":
            Xtf_li,ytf_li = self.preprocessor_li.fit_transform(X,y)
            Xtf_li = self.postprocessor_li.fit_transform(Xtf_li)
            print "Training.."
            map(lambda x:x.fit(Xtf_li,ytf_li),self.clfs)
            ner_trained_model = {
                "preprocessor" : self.preprocessor_li,
                "postprocessor" : self.postprocessor_li,
                "clfs" : self.clfs,
            }
            dump_model(ner_trained_model,"./Resources/li_model.pkl")
            print "Dumped LI model at `./Resources/li_model.pkl``"
        elif self.mode == "test":
            if len(X[0]) == 0: # exception
                return []
            X_tf_li = self.preprocessor_li.transform(X,y)
            X_tf_li = self.postprocessor_li.transform(X_tf_li)
            ypred_lists = [clf.predict(X_tf_li) for clf in self.clfs]
            y_pred = perform_or(ypred_lists)        #   corrected majority voting - pending
            y_pred_str = [self.inv_label_dict_li[elem] for elem in y_pred]
            return y_pred_str

def run_testing():
    ids,utters = get_data("./TestingData/Input1.txt")
    ids = ids
    utters = utters
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
    ner_annotator = Sklearn_ner(mode="test")
    y_ne = [ner_annotator.run([X_ne_str[i]]) for i in xrange(len(X_ne_str))]
    # Deconstruct NE's
    print "deconstructing ne's"
    X_li,ne_bak,_ = deconstructor(X_ne,y_ne,"NE")
    X_li_str = [' '.join(x_li) for x_li in X_li]
    # Annotating Lang
    print "annotating languages"
    li_annotator = Sklearn_li(mode="test")
    y_li =[li_annotator.run([X_li_str[i]]) for i in xrange(len(X_li_str))]
    # Reconstructing NE's
    print "reconstructing ne's"
    y_li_recon_ne = reconstructor(y_li,ne_bak,"NE")
    # Reconstructing X's
    print "reconstructing x's"
    y_li_ne_recon_x = reconstructor(y_li_recon_ne,xnx_bak,"X")
    # a final test to check reconstruction
    print "running tests"
    run_tests(include_ner_subcat=False)
    pprint(annotation_count_test("./TestingData/Annotation1.txt"))
    prepare_submission(y_ids,y_li_ne_recon_x)

def run_training(add_extra_datasets=True):
    ids,X,y = get_training_data("Input1.txt","Annotation1.txt",NUM_UTTERANCES)
    if add_extra_datasets:
        ids_rcv,X_rcv,y_rcv = get_training_data("extraTraining_Input.txt","extraTraining_Annotation.txt",NUM_UTTERANCES_EXTRA)
        X=X+X_rcv
        y=y+y_rcv
    Sklearn_li(mode="train").run(X,y)
    Sklearn_ner(mode="train").run(X,y)

def main(mode="validation"):
    if mode == "validation":
        print "Validating LI"
        run_validation("li",add_extra_datasets=True)
        print "Validating NER"
        run_validation("ne",add_extra_datasets=True)
    elif mode == "training":
        print "Training..."
        run_training(add_extra_datasets=True)
    elif mode == "testing":
        print "Testing..."
        run_testing()


if __name__ == '__main__':
    # main("validation")
    main("training")
    main("testing")