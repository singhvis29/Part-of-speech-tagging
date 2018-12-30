# Part-of-speech-tagging
Part of Speech tagging using Naive Bayes Classifier, Viterbi Algorithm and Gibbs Sampling

### Data
A large corpus of labeled training and testing data, consisting of nearly 1 million words and 50,000 sentences. Each line consists of a word, followed by a space, followed by one of 12 part-of-speech tags: ADJ (adjective), ADV (adverb), ADP (adposition), CONJ (conjunction), DET (determiner), NOUN, NUM (number), PRON (pronoun), PRT (particle), VERB, X (foreign word), and . (punctuation mark). Sentence boundaries are indicated by blank lines.

### Files
**1.** bc.train: trainig data consisting of word followed by it POS label on which the prior, likelihood, and joint probabilities are calculated<br>
**2.** bc.test: testing data in same format as the training file on which the algorithms are run to determine the Part of speech<br>
**3.** bc.test.tiny: Testing file with 4 sentences in similar format as the training file
**4.** label.py pos_scorer.py: Boilerplate programs for running the algorithms and calculating joint probabilities and accuracy
**5.** pos_solver: Python program which contains codes for the Naive Bayes, Viterbi and Gibbs Sampling algorithm

### Implementation


