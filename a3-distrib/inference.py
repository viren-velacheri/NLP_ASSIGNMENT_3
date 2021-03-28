# inference.py

from models import *
from treedata import *
from utils import *
from collections import Counter
from typing import List

import copy

import numpy as np


def decode_bad_tagging_model(model: BadTaggingModel, sentence: List[str]) -> List[str]:
    """
    :param sentence: the sequence of words to tag
    :return: the list of tags, which must match the length of the sentence
    """
    pred_tags = []
    for word in sentence:
        if word in model.words_to_tag_counters:
            pred_tags.append(model.words_to_tag_counters[word].most_common(1)[0][0])
        else:
            pred_tags.append("NN") # unks are often NN
    return labeled_sent_from_words_tags(sentence, pred_tags)


def viterbi_decode(model: HmmTaggingModel, sentence: List[str]) -> LabeledSentence:
    """
    :param model: the HmmTaggingModel to use (wraps initial, emission, and transition scores)
    :param sentence: the words to tag
    :return: a LabeledSentence containing the model's predictions. See BadTaggingModel for an example.
    """
    numWords = len(sentence)
    numStates = model.init_log_probs.size
    viterbi_table = np.zeros((numStates, numWords))
    pi_vals = np.zeros((numStates, numWords))
    top_path = np.zeros(numWords).astype(np.int32)
    wordIndexer = model.word_indexer
    tagIndexer = model.tag_indexer



    y = 0
    while y < numStates:
        indexToAdd = wordIndexer.index_of(sentence[0])
        if indexToAdd == -1:
            indexToAdd = 0
        viterbi_table[y, 0] = model.init_log_probs[y] + model.emission_log_probs[y, indexToAdd]
        y += 1
    
    for i in range(1, numWords):
        for j in range(0, numStates):
            temp_sum = model.transition_log_probs[:, j] + viterbi_table[:, i - 1]
            indexToAdd = wordIndexer.index_of(sentence[i])
            if indexToAdd == -1:
                indexToAdd = 0
            viterbi_table[j, i] = np.max(temp_sum) + model.emission_log_probs[j, indexToAdd]
            pi_vals[j, i - 1] = np.argmax(temp_sum)
    
    for k in range(0, numStates):
        viterbi_table[k, numWords - 1] = viterbi_table[k, numWords - 1] + model.transition_log_probs[k, tagIndexer.index_of("STOP")]

    top_path[-1] = np.argmax(viterbi_table[:, -1])    

    for n in range(numWords - 2, -1, -1):
        top_path[n] = pi_vals[int(top_path[n + 1]), n]
    
    pred_tags = []
    
    for k in range(numWords):
        pred_tags.append(tagIndexer.get_object(top_path[k]))
    
    return labeled_sent_from_words_tags(sentence, pred_tags)
    
    
    



def beam_decode(model: HmmTaggingModel, sentence: List[str], beam_size: int) -> LabeledSentence:
    """
    :param model: the HmmTaggingModel to use (wraps initial, emission, and transition scores)
    :param sentence: the words to tag
    :param beam_size: the beam size to use
    :return: a LabeledSentence containing the model's predictions. See BadTaggingModel for an example.
    """



    numWords = len(sentence)
    numStates = model.init_log_probs.size
    # viterbi_table = np.zeros((numStates, numWords))
    # pi_vals = np.zeros((numStates, numWords))
    # top_path = np.zeros(numWords).astype(np.int32)
    wordIndexer = model.word_indexer
    tagIndexer = model.tag_indexer

    y = 0
    tempBeam = Beam(beam_size)
    while y < numStates:
        indexToAdd = wordIndexer.index_of(sentence[0])
        if indexToAdd == -1:
            indexToAdd = 0
        score = model.init_log_probs[y] + model.emission_log_probs[y, indexToAdd]
        tempList = []
        tempList.append(y)
        tempBeam.add(tempList, score)
        y += 1
    
    for i in range(1, numWords):
        # oldBeam = copy.deepcopy(tempBeam)
        temp2 = Beam(beam_size)
        for j in range(beam_size):
            tagList = tempBeam.elts[j]
            prevTag = tagList[-1]
            prevScore = tempBeam.scores[j]
            for k in range(numStates):
                indexToAdd = wordIndexer.index_of(sentence[i])
                if indexToAdd == -1:
                    indexToAdd = 0
                score = model.transition_log_probs[prevTag, k] + model.emission_log_probs[k, indexToAdd] + prevScore
                tempList = tagList[:]
                tempList.append(k)
                # print(tempList)
                temp2.add(tempList, score)
        tempBeam = copy.deepcopy(temp2)
        # print(tempBeam.get_elts())


    # tempBeam2 = Beam(beam_size)
    # for beam in range(beam_size):
    #     tagList = tempBeam.elts[beam]
    #     prevTag = tagList[-1]
    #     prevScore = tempBeam.scores[beam]
    #     score = prevScore + model.transition_log_probs[prevTag, tagIndexer.index_of("STOP")]
    #     tempBeam2.add(tagList, score)
    # tempBeam = copy.deepcopy(tempBeam2)
    

    topList = tempBeam.head()
    # print(topList)
    # if len(topList) != numWords:
    #     print(len(topList))
    #     print("\n")
    #     print(numWords)
    pred_tags = []
    for tag in range(len(topList)):
        pred_tags.append(tagIndexer.get_object(topList[tag]))
    
    return labeled_sent_from_words_tags(sentence, pred_tags)

        # for j in range(0, numStates):
        #     temp_sum = model.transition_log_probs[:, j] + viterbi_table[:, i - 1]
        #     indexToAdd = wordIndexer.index_of(sentence[i])
        #     if indexToAdd == -1:
        #         indexToAdd = 0
        #     viterbi_table[j, i] = np.max(temp_sum) + model.emission_log_probs[j, indexToAdd]
        #     pi_vals[j, i - 1] = np.argmax(temp_sum)
    
