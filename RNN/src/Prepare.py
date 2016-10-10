'''
Created on Jul 17, 2016

@author: Veltzer Doron
'''

import numpy
from distutils.util import strtobool

import TurkishPhonology

def main():
    # generate training input
    # Load the TELL lexicon dataset
    trainData = numpy.loadtxt(fname = 'csv/data.csv', delimiter = ',', skiprows = 1, dtype = numpy.str)
    
    FEATURES = 'FEATURES'
    FULL = ''
    TRAIN_NAME = 'csv/train' + FEATURES + FULL + '.csv'
    TEST_NAME = 'csv/test' + FEATURES + FULL + '.csv'
    
    if (FULL == 'FULL'):
        # use stems that are less than MAX_MORAS moras long
        MAX_MORAS = 5
        legalStructures = numpy.array(map(int, trainData[:, 3])) <= MAX_MORAS
    else:
        # or only prosodic structures of the forms CVC CVCC CVCVC
        structures = numpy.asanyarray(trainData[:, 5])
        legalStructures = (structures == 'CVC') | (structures == 'CVCC') | (structures == 'CVCVC')
    
    words = [unicode(word, 'utf-8') for word in trainData[legalStructures, 0]]
    speakers = trainData[legalStructures, 4]
    
    maxPhonemes = max([len(word) for word in words])
    
    #load the alternation column
    alternations = numpy.array(map(strtobool, trainData[legalStructures, 2])).reshape(1, -1)
    
    # handle duplicates
    duplicateDiffSpeaker = 0
    duplicateSameSpeaker = 0
    variationsDiffSpeaker = 0
    variationsSameSpeaker = 0
    variationsBothSpeaker = 0
    bothVariatingAndNoneCounter = 0
    uniqueWords = []
    uniqueAlternations = []
    
    for i in xrange(len(words)):
        wordi = words[i]
        alternationi = alternations[0, i]
        if not wordi in uniqueWords:
            bothVariatingAndNone = False
            variationsSameSpeakerFlag = False
            variationsDiffSpeakerFlag = False
            noneDiffSpeakerFlag = False
            variating = 0
            noning = 0
            for j in xrange(i + 1, len(words)):
                wordj = words[j]
                alternationj = alternations[0, j]
                if (wordi == wordj):
                    if (speakers[i] == speakers[j]):
                        if (alternationi != alternationj):
                            variating = variating + 1 
                            variationsSameSpeaker = variationsSameSpeaker + 1
                            variationsSameSpeakerFlag = True
                            if not bothVariatingAndNone:
                                bothVariatingAndNoneCounter = bothVariatingAndNoneCounter + 1
                                bothVariatingAndNone = True
                        else:
                            noning = noning + 1
                        duplicateSameSpeaker = duplicateSameSpeaker + 1
                    else:
                        if (alternationi != alternationj):
                            variating = variating + 1 
                            variationsDiffSpeaker = variationsDiffSpeaker + 1
                            variationsDiffSpeakerFlag = True
                            if not bothVariatingAndNone:
                                bothVariatingAndNoneCounter = bothVariatingAndNoneCounter + 1
                                bothVariatingAndNone = True
                        else:
                            noneDiffSpeakerFlag = True
                            noning = noning + 1
                        duplicateDiffSpeaker = duplicateDiffSpeaker + 1
            if bothVariatingAndNone:
                uniqueWords.append(wordi)
                if variating >= noning:
                    uniqueAlternations.append(alternationi)
                else:
                    uniqueAlternations.append(not alternationi)
                '''
                uniqueWords.append(wordi)
                uniqueAlternations.append(True)
                uniqueWords.append(wordi)
                uniqueAlternations.append(False)
                '''
            else:
                uniqueWords.append(wordi)
                uniqueAlternations.append(alternationi)
            if variationsSameSpeakerFlag and variationsDiffSpeakerFlag and noneDiffSpeakerFlag :
                variationsBothSpeaker = variationsBothSpeaker + 1
    
    print(duplicateDiffSpeaker)
    print(duplicateSameSpeaker)
    print(variationsDiffSpeaker)
    print(variationsSameSpeaker)
    print(variationsBothSpeaker)
    print(bothVariatingAndNoneCounter)
    print(len(uniqueWords))
    print(numpy.sum(uniqueAlternations))
    print(numpy.sum(legalStructures))
    
    # build the words' input and output to the network (features or IDs)
    IDs = numpy.zeros(shape = [len(uniqueWords), maxPhonemes])
    outputFeatures = numpy.zeros(shape = [len(uniqueWords), TurkishPhonology.featuresNum])
    for i in xrange(len(uniqueWords)):
        # extract id of the unique word's phonemes
        word = uniqueWords[i]
        lenWord = len(word)
        for j in xrange(lenWord):
            phoneme = word[j]
            
            # fillers to the right
            #IDs[i, j] = TurkishPhonology.phoneme2Index(phoneme)
            
            # fillers to the left
            IDs[i, maxPhonemes - lenWord + j] = TurkishPhonology.phoneme2Index(phoneme)
        
        # generate its alternating phoneme's features
        outputFeatures[i, :] = TurkishPhonology.phoneme2Features(word[-1]) - 1
        outputFeatures[i, TurkishPhonology.featuresNum - 1] += uniqueAlternations[i]
    
    XY = numpy.concatenate((IDs, outputFeatures), 1)
    
    numpy.savetxt(TRAIN_NAME, delimiter = ',', X = XY, fmt = '%d')
    
    # generate nonce input
    # load the nonce words
    testData = numpy.loadtxt(fname = 'csv/nonce.csv', delimiter = ',', skiprows = 1, dtype = numpy.str)
    nonceWords = [unicode(word, 'utf-8') for word in testData[:]]
    
    '''
    # generate all nonce options (this is way too loose)
    # generate CVC stems
    for coda in TurkishPhonology.getConsonants():
        for vowel in TurkishPhonology.getVowels():
            for final in TurkishPhonology.finalConsonants:
                nonceWords.append(coda + vowel + final)
    
    # generate CVCVC stems
    # add front VH stems 
    for coda1 in TurkishPhonology.getConsonants():
        for vowel1 in TurkishPhonology.getVowels('Front'):
            for coda2 in TurkishPhonology.getConsonants():
                for vowel2 in TurkishPhonology.getVowels('Front'):
                    for final in TurkishPhonology.finalConsonants:
                        nonceWords.append(coda1 + vowel1 + coda2 + vowel2 + final)
    
    # add back VH stems 
    for coda1 in TurkishPhonology.getConsonants():
        for vowel1 in TurkishPhonology.getVowels('Back'):
            for coda2 in TurkishPhonology.getConsonants():
                for vowel2 in TurkishPhonology.getVowels('Back'):
                    for final in TurkishPhonology.finalConsonants:
                        nonceWords.append(coda1 + vowel1 + coda2 + vowel2 + final)
    '''
    
    # build the words' input to the network (features or IDs)
    nonceIDs = numpy.zeros(shape = [len(nonceWords), maxPhonemes])
    for i in xrange(len(nonceWords)):
        nonceWord = nonceWords[i]
        lenWord = len(nonceWord)
        for j in xrange(lenWord):
            phoneme = nonceWord[j]
            
            # fillers to the right
            #nonceIDs[i, j] = TurkishPhonology.phoneme2Index(phoneme)
            
            # fillers to the left
            nonceIDs[i, maxPhonemes - lenWord + j] = TurkishPhonology.phoneme2Index(phoneme)
    
    nonceXY = nonceIDs
    
    numpy.savetxt(TEST_NAME, delimiter = ',', X = nonceXY, fmt = "%d")

if __name__ == "__main__":
    main()