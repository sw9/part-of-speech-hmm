# Hidden Markov Models for Part of Speech Tagging

This was an assignment for the AI class at Penn. It contains code that trains a hidden markov model on the Brown Corpus. When given a list of tokens (which forms a sentence), the hmm will return the most likely part of speech for each token in the sentence. In this model, the hidden states are the parts of speech and the observations are the words or punctuation in a sentence.

## Dependencies ##
Python 2.x

## Use

### load_corpus(path) ###

When given a string representing the path to the corpus, this function reads it in and returns a list of list of tuples where each list represents a separate sentence. Each tuple represents a word or punctuation mark in the sentence and its labeled part of speech.

### Tagger class ###

Initialize an object of this class with the list of sentences returned by `load_corpus`.

    >> c = load_corpus("brown_corpus.txt")
    >> t = Tagger(c)
    
### Most probable tags

Given a sequence of tokens, the `Tagger` class can return the most common part of speech for each token. This is the simple approach to POS that doesn't make use of the hmm.
    
    >> s = "I am waiting to reply".split()
    >> t.most_probable_tags(s)
    ['PRON', 'VERB', 'VERB', 'PRT', 'NOUN']
    
### Viterbi Algorithm

The `Tagger` class can also return the most probable part of speech to a sequence of tokens using the Viterbi algorithm. This is the more sophisticated approach that takes into account the probability of transitioning between different parts of speech and the probability of observing some token while in a part-of-speech state.

    >> t.viterbi_tags(s)
    ['PRON', 'VERB', 'VERB', 'PRT', 'VERB']
