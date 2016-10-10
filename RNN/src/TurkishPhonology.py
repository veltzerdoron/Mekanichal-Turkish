# -*- coding: utf-8 -*-

'''
Created on Aug 22, 2016

@author: veltzer
'''

'''
The following is based on Kamil Stachowsky's work,
A phonological encoding of Turkish for neural networks (2012)

Feature    Labial        Alveolar       Post-alveolar    Palatal    Guttural
Stop       p b m         t d n                           c ɟ -      k g -    1
Affricate                               č ǯ -            h - -               2
Fricative  f v -         s z -          š ž -                                3
Liquid                   - ɾ ɫ          - r l            - j -               4
HighVowelIndex vowel                                               - i y      - ɯ u    5
Low vowel                                                - e ø      - a o    6
           1             2              3                4          5
'''

# how many features per phoneme
featuresNum = 3

# maps phonemes to their features
featuresMap = {
    unicode('p', 'utf-8'): [1, 1, 1],
    unicode('b', 'utf-8'): [1, 1, 2],
    unicode('m', 'utf-8'): [1, 1, 3],
    unicode('f', 'utf-8'): [1, 3, 1],
    unicode('v', 'utf-8'): [1, 3, 2],
    unicode('t', 'utf-8'): [2, 1, 1],
    unicode('d', 'utf-8'): [2, 1, 2],
    unicode('n', 'utf-8'): [2, 1, 3],
    unicode('s', 'utf-8'): [2, 3, 1],
    unicode('z', 'utf-8'): [2, 3, 2],
    unicode('ɾ', 'utf-8'): [2, 4, 2],
    unicode('ɫ', 'utf-8'): [2, 4, 3],
    unicode('č', 'utf-8'): [3, 2, 1],
    unicode('ǯ', 'utf-8'): [3, 2, 2],
    unicode('š', 'utf-8'): [3, 3, 1],
    unicode('ž', 'utf-8'): [3, 3, 2],
    unicode('r', 'utf-8'): [3, 4, 2],
    unicode('l', 'utf-8'): [3, 4, 3],
    unicode('c', 'utf-8'): [4, 1, 1],
    unicode('ɟ', 'utf-8'): [4, 1, 2],
    unicode('j', 'utf-8'): [4, 4, 2],
    unicode('i', 'utf-8'): [4, 5, 2],
    unicode('y', 'utf-8'): [4, 5, 3],
    unicode('e', 'utf-8'): [4, 6, 2],
    unicode('ø', 'utf-8'): [4, 6, 3],
    unicode('k', 'utf-8'): [5, 1, 1],
    unicode('g', 'utf-8'): [5, 1, 2],
    unicode('h', 'utf-8'): [5, 2, 1],
    unicode('ɯ', 'utf-8'): [5, 5, 2],
    unicode('u', 'utf-8'): [5, 5, 3],
    unicode('a', 'utf-8'): [5, 6, 2],
    unicode('o', 'utf-8'): [5, 6, 3]
}

# maps phonemes to their index
indexMap = {
    unicode('p', 'utf-8'): 1,
    unicode('b', 'utf-8'): 2,
    unicode('m', 'utf-8'): 3,
    unicode('f', 'utf-8'): 4,
    unicode('v', 'utf-8'): 5,
    unicode('t', 'utf-8'): 6,
    unicode('d', 'utf-8'): 7,
    unicode('n', 'utf-8'): 8,
    unicode('s', 'utf-8'): 9,
    unicode('z', 'utf-8'): 10,
    unicode('ɾ', 'utf-8'): 11,
    unicode('ɫ', 'utf-8'): 12,
    unicode('č', 'utf-8'): 13,
    unicode('ǯ', 'utf-8'): 14,
    unicode('š', 'utf-8'): 15,
    unicode('ž', 'utf-8'): 16,
    unicode('r', 'utf-8'): 17,
    unicode('l', 'utf-8'): 18,
    unicode('c', 'utf-8'): 19,
    unicode('ɟ', 'utf-8'): 20,
    unicode('j', 'utf-8'): 21,
    unicode('i', 'utf-8'): 22,
    unicode('y', 'utf-8'): 23,
    unicode('e', 'utf-8'): 24,
    unicode('ø', 'utf-8'): 25,
    unicode('k', 'utf-8'): 26,
    unicode('g', 'utf-8'): 27,
    unicode('h', 'utf-8'): 28,
    unicode('ɯ', 'utf-8'): 29,
    unicode('u', 'utf-8'): 30,
    unicode('a', 'utf-8'): 31,
    unicode('o', 'utf-8'): 32
}

# indicates a double prosodic position for the previous vowel
longVowel = unicode(':', 'utf-8')

# the number of phonemes in the language
maxIndex =  max(indexMap.values())

# get all the consonants in the language
def getConsonants():
    consonants = []
    for k,v in featuresMap.items():
        if (v[1] < 5):
            consonants.append(k)
    return consonants

# get all the features of vowels in the language according to required spec:
#    'High'    : high vowels
#    'Low'     : low vowels
#    'Front'   : front vowels
#    'Back'    : back vowels
#    'Round'   : round vowels
#    'UnRound' : unround vowels
def getVowels(spec = None):
    vowels = []
    for k,v in featuresMap.items():
        if (spec is None):
            # return array of all vowels
            if (v[1] >= 5):
                vowels.append(k)
        if (spec == 'High'):
            # return array of all high vowels
            if (v[1] == 5):
                vowels.append(k)
        if (spec == 'Low'):
            if (v[1] == 6):
                vowels.append(k)
        if (spec == 'Front'):
            # return array of all high vowels
            if ((v[1] >= 5) and (v[0] == 4)):
                vowels.append(k)
        if (spec == 'Back'):
            if ((v[1] >= 5) and (v[0] == 5)):
                vowels.append(k)
        if (spec == 'Round'):
            # return array of all high vowels
            if ((v[1] >= 5) and (v[2] == 3)):
                vowels.append(k)
        if (spec == 'UnRound'):
            if ((v[1] >= 5) and (v[2] == 2)):
                vowels.append(k)
    return vowels

# define the final consonants of interest
finalConsonants = [unicode(c, 'utf-8') for c in ['k', 'č', 't', 'p']]
finalConsonantIndices = [indexMap[x] for x in finalConsonants]

# translate a phoneme to its feature set (also handles long vowels)
# save previous phoneme in case this is a long vowel
previousPhoneme = None
def phoneme2Features(x):
    global previousPhoneme
    #: means the previous x was a vowel and is now declared long, return its features again
    if (x == longVowel):
        if (previousPhoneme is None): 
            raise Exception('illegal word')
        return featuresMap[previousPhoneme]
    #otherwise remember the previous x and return its features
    previousPhoneme = x
    return featuresMap[x]

# translate a phoneme to its index (also handles long vowels)
def phoneme2Index(x):
    global previousPhoneme
    #: means the previous x was a vowel and is now declared long, return its index again
    if (x == longVowel):
        if (previousPhoneme is None): 
            raise Exception('illegal word')
        return indexMap[previousPhoneme]
    #otherwise remember the previous x and return its index
    previousPhoneme = x
    return indexMap[x]

# receives a list of indices and returns the length of the word
def lengthIndices(indices):
    result = 0
    for index in indices:
        if (index > 0):
            result = result + 1
    return result

# Point Of Articulation for the final consonant phoneme with the given index
def POAIndex(index):
    return finalConsonantIndices.index(index)

# Point Of Articulation for the final consonant phoneme with the given char representation
def POA(c):
    return POAIndex(indexMap[unicode(c, 'utf-8')])

#vowel 
highVowels = getVowels('High')
backVowels = getVowels('Back')

def HighVowel(v):
    return (v in highVowels)

def BackVowel(v):
    return (v in backVowels)

highVowelIndices = [indexMap[v] for v in highVowels]
backVowelIndices = [indexMap[v] for v in backVowels]

def HighVowelIndex(i):
    return (i in highVowelIndices)

def BackVowelIndex(i):
    return (i in backVowelIndices)

# To translate the phonemes of word to features perform the following
#features = [[phoneme2Features(phoneme) for phoneme in unicode(word, "utf-8")] for word in words]
# To translate the phonemes of word to IDs perform the following
#indices = [[phoneme2Index(phoneme) for phoneme in unicode(word, "utf-8")] for word in words]