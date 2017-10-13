This is an assignment of our Natural Language Processing Course.
It can identify the language of a given text using a series of trained language models.
In this a assignment, we first build a series of character language models based on a series of text files in different languages in the training set. Then we identify the language of each text file in the development set and test set.

Team Members: [ 'Shufan(Atom) Cai', 'Dinghao(Kevin) Liu']

--------------------------
  How to Run the Program
--------------------------

# Usage

python langid.py --SMOOTHING_TYPE --TEST_OBJ


# Available Command Line Arguments:

--SMOOTHING_TYPE	Indicate which smoothing type to use, which has four options:
			--unsmoothed
			--laplace
			--nltk-kneser-ney	# Use NLTK's original Kneser-Ney smoothing functions
			--improved-kneser-ney	# Use the improved Kneser-Ney smoothing function implemented by ourself, 
							which is based on the principle in our tutorial's chapter 4
							and can solve the problem of assigning zero probabilities to unseen ngrams.   

--TEST_OBJ		Indicate which data set to use for the test, which has two options:
			--dev	# Test the development set of files
			--tes	# Test the test set of files


# Program Usage Examples

python langid.py --unsmoothed --dev		# Test the dev set using unsmoothed models
python langid.py --laplace --tes		# Test the test set using Laplace smoothing
python langid.py --nltk-kneser-ney --tes	# Test the test set using NLTK's original Kneser-Ney smoothing functions
python langid.py --improved-kneser-ney --dev	# Test the dev set using our self-implemented improved Kneser-Ney smoothing function


-----------------------------
  About Tunning Parameter n
-----------------------------

# The Result of Tunning Parameter n

Unsmoothed Model
{
	The Tuned Parameter n: 2
	Max Match Rate: 5 / 55
	Best Match Average Perplexity: 6.76094554588
}

Laplace Smoothing Model
{
	The Tuned n Value: 3
	Max Match Rate: 54 / 55
	Best Match Average Perplexity: 9.06971102989
}


# How We Tuned Parameter n

We find the optimum n value by setting different n values for each type of model to get the maximum match rate and the minimum average perplexity.
First, we built each type of model with different n values ranging from 2 to 20. 
Then we compared the matching results of the models using the same smoothing method but different n values.
We compared the matching results in the following way:
 - The result with higher match rate is better.
 - If two results have the same match rate, the one with lower average perplexity of matched files is better.
In this way, we finally got the best n value for each type of model that can let the model get the best matching result.


# How to Use the Program to Tune Parameter n

The program can help us find the best n values automatically.
To let the program do so you just need to run the program with the --SMOOTHING_TYPE cmd line argument as --unsmoothed or --laplace, and the --TEST_OBJ cmd line argument as "dev".
Once running, the program allows you to input the min and max values of n which indicate the range of n value when finding the optimum n value.


--------------------
	P.S.
--------------------
1. We've included the data set files in our compressed folder considering that the size of them is not big and that doing so can make it convenient to test the program. You can also reset the root paths of data set files at the top part of the langid.py file.

2. The "results_dev_kneser-ney.txt" and "results_test_kneser-ney.txt" files are the results derived from our improved Kneser-Ney smoothing method rather than NLTK's original Kneser-Ney smoothing method. We choose our improved Kneser-Ney smoothing function to output the result because its result is better than that of NLTK's original Kneser-Ney smoothing method.