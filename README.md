# Part-of-speech-tagging
Part of Speech tagging using Naive Bayes Classifier, Viterbi Algorithm and Gibbs Sampling

### Data
A large corpus of labeled training and testing data, consisting of nearly 1 million words and 50,000 sentences. Each line consists of a word, followed by a space, followed by one of 12 part-of-speech tags: ADJ (adjective), ADV (adverb), ADP (adposition), CONJ (conjunction), DET (determiner), NOUN, NUM (number), PRON (pronoun), PRT (particle), VERB, X (foreign word), and . (punctuation mark). Sentence boundaries are indicated by blank lines.

### Files
**1.** bc.train: trainig data consisting of word followed by it POS label on which the prior, likelihood, and joint probabilities are calculated<br>
**2.** bc.test: testing data in same format as the training file on which the algorithms are run to determine the Part of speech<br>
**3.** bc.test.tiny: Testing file with 4 sentences in similar format as the training file<br>
**4.** label.py pos_scorer.py: Boilerplate programs for running the algorithms and calculating joint probabilities and accuracy<br>
**5.** pos_solver: Python program which contains codes for the Naive Bayes, Viterbi and Gibbs Sampling algorithm<br>

### Implementation

#### Assumptions:
![alt text](https://github.com/singhvis29/Part-of-speech-tagging/blob/master/BayesNets.JPG)


#### ALGORITHMS

**NAIVE BAYES(SIMPLE)**<br>
This is implemented by maximising the posterior probability(check posterior
probability section) for each tag given a word. In case the test data has an
unseen word. A small probability is substituted for P(Word|Tag)

**VITERBI(HMM)**<br>
The algorithm is implemented by initializing two  dictionary of dictionaries
'viterbi' and 'backtrack' the keys of outer dictionary are the unique tags 
and the keys of the inner dict are the words in the sentence. The viterbi dict
stores the max score obtained by implementing the viterbi algorithm at a 
particular state and the backtrack dict stores the value of tags from which
the maximum score is obtained(argmax). To generate a sequence the tag is obtained
from the viterbi dict whose probability is maximum for the last word, this is 
the last tag of the given sentence. corresponding values is obtained from 
backtrack dict, this is the tag of the the second last word. The tags are traced
back from the backtrack dict. This order is then reversed to obtain the sequence
of tags for the given sentence. If in test a word does not appear for a given
tag then it is replaced by a samll probability

**MCMC(COMPLEX)**<br>
For a given sentence a random sequence of tags is initialized, from this sequence
for each position, starting from first tag, the values are changed to each tag
and the corresponding joint probability is calculated using 'joint_probability'
function keeping all the other tags fixed. Then a random tag is fixed at the 
position using the function random.choice, then same step is performed for all
the tags in the tag sequence, this way one sample is generated. Number of iterations
and burning iterations(samples to discard initially) are chosen, this creates
number of samples from the given probability distribution. To find out the sequence
from the samples, we have chosen the tag appearing the most number of times at each
position

#### Posterior probabilities

**[Simple]**<br>
The posterior probability is calculated using Naive Bayes assumption, that
each hidden variable is independent from every other hidden variable. Given a 
set of observed variables(words) and hidden variables (POS tags), for each word 
and corresponding tag, the posterior probability of each tag given a word is given by 
P(tag|word) = (P(word|tag) * P(tag))/P(word)
We can ignore the denominator as we have to compare the score for each tag and
consider the maximum.
 i.e. P(tag|word) α (P(word|tag) * P(tag))
This is for one word and corresponding tag, for a given sequence, we have to 
multiply the posterior probabilities of all the word-tag combination
P(tag1|word1)*P(tag2|word2)…P(tagn|wordn) α (P(word1|tag1) * P(tag1)) * 
                       (P(word2|tag2) * P(tag2)) ….. (P(wordn|tagn) * P(tagn))
 Also, since the probabilities are stored in logs, we convert the above equation
 to the following by taking logs on both sides
Posterior probability of the given word-tag sequence = log(P(tag1|word1)) + 
                                       log(P(tag2|word2))…+log(P(tagn|wordn))
In the program the probability P(word|tag) is the emission probability(likelihood)
 and is stored in the variable ‘pws’ and the the probability P(tag) is the probability
 of the hidden variable(prior) and is stores in the variable ‘ps’

**[HMM]**<br>
The probability of each element in a Hidden Markov Chain depend on the previous
state(transition probability) and the probability of each word(emission probability)
depends on the hidden state. The probability of a given sequence of words and 
hidden states(tags) is given by
Probaility = [P(tag1)*P(word1|tag1)]* [P(word2|tag2)*P(tag2|tag1)]….* 
                                               [P(wordn|tagn)*P(tagn|tagn-1)]
The transition probabilities (e.g. P(tag2|tag1)) are stored in pss

**[Complex]**<br>
According to figure 1c given in the question, the probability of state depends
 on the two previous states(complex probability) and the probability of each 
word(emission probability) depends on the hidden state. The probability of a 
given sequence of words is given by
Probability = [P(tag1)*P(word1|tag1)]* [P(word2|tag2)*P(tag2|tag1)]*
       [P(word3|tag3)*P(tag3|tag2,tag1)]…* [P(wordn|tagn)*P(tagn|tagn-1,tagn-2)]
The complex probabilities (e.g. P(tag3|tag1,tag2))  are stored in ‘pc’
Please note that ‘pc’ stores the probabilities that three tags occur in a sequence.




