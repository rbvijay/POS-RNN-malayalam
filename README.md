POS-RNN-malayalam
=================

POS tagging using Recurrent Neural Network for Malayalam language

1. Was able to achive 74% accuracy with RNN on test corpus (740/1005 correct predictions).
2. Input features for RNN were words, and preceding POS tags only, since test corpus can have preceding pos available but not succeeding tags.
3. Output as POS tags.
4. Two methods were used for unseen words - word similarity based function, words ending with similar letters.
5. Relavant comments are avaiable in source code where necessary. 
6. Output is available in op.txt in format ===> word : predicted pos : actual pos
7. Project can be imported in eclipse directly.
8. NLP_POS_RNN\src\com\pos\util\PosTagger.java contains executable code to run test on corpus.
