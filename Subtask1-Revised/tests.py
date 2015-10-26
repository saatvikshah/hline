__author__ = 'tangy'

from batchio import get_data

"""
Module for running tests to verify the authenticity of generated annotation files
"""

ALLOWED_LABELS_WITH_NERSUB = ["en","hi","bn","ta","gu","mr","kn","te","ml","O","NE_P","NE_L","NE_O","X","MIX"]
ALLOWED_LABELS_WITHOUT_NERSUB = ["en","hi","bn","ta","gu","mr","kn","te","ml","O","NE","X","MIX"]

def run_tests(include_ner_subcat = True):
    ids,utters = get_data("./TestingData/Input1.txt")
    ids,labels = get_data("./TestingData/Annotation1.txt")
    if include_ner_subcat:
        ids,labels_ner = get_data("./TestingData/Ann_NERSub.txt")
    for idx in xrange(len(ids)):
        utter_idx = utters[idx].split()
        labels_idx = labels[idx].split()
        if include_ner_subcat:
            labels_ner_idx = labels_ner[idx].split()
        assert len(utter_idx) == len(labels_idx),"mismatch in length of label_length/utter_length in Annotation1.txt at utter %d" % ids[idx]
        if include_ner_subcat:
            assert len(utter_idx) == len(labels_ner_idx),"mismatch in length of label_length/utter_length in Ann_NERSub.txt at utter %d" % ids[idx]
        for label in labels_idx:
            assert label in ALLOWED_LABELS_WITHOUT_NERSUB, "some invalid label %s found in Annotation1.txt at utter %d" % (label,ids[idx])
        if include_ner_subcat:
            for label in labels_ner_idx:
                assert label in ALLOWED_LABELS_WITH_NERSUB, "some invalid label %s found in Ann_NERSub.txt at utter %d" % (label,ids[idx])
            for sub_idx in xrange(len(labels_idx)):
                if labels_idx[sub_idx] == "NE": assert labels_ner_idx[sub_idx].startswith("NE") or labels_ner_idx[sub_idx] == "en","some invalid label %s found in Ann_NERSub.txt at utter %d" % (labels_idx[sub_idx],ids[idx])


def annotation_count_test(fname):
    op = open(fname,'r')
    word_list = op.read().split()
    annotation = []
    annotation_dict = {}
    for word in word_list:
        if not(word.startswith("<") or word.startswith("id")) and not(word in annotation):
            annotation.append(word)
            annotation_dict[word] = 1
        elif word in annotation:
            annotation_dict[word] += 1
    return annotation_dict

