from __future__ import division

import os
import nltk
import math
import sys

# Global Constants

""" Root Paths of Data Files """
TRA_SET_ROOT = "811_a1_train/"		# The root path of training set files, all the training files should be under this directory.
DEV_SET_ROOT = "811_a1_dev/"	# The root path of dev set files, all the dev files should be under this directory.
TES_SET_ROOT = "811_a1_test_final/"		# The root path of test set files, all the test files should be under this directory.

NO_SMOOTHING_CMD = "--unsmoothed"
ADD_ONE_CMD = "--laplace"
NLTK_KNESER_NEY_CMD = "--nltk-kneser-ney"
IMPROVED_KNESER_NEY_CMD = "--improved-kneser-ney"

TES_SET_CMD = "--tes"
DEV_SET_CMD = "--dev"

TRA_FILE_SUFFIX = ".tra"
DEV_FILE_SUFFIX = ".dev"
TES_FILE_SUFFIX = ".tes"

DEV_NO_SMOOTHING_RESULTS_OUTPUT_FILENAME = "results_dev_unsmoothed.txt"
DEV_ADD_ONE_RESULTS_OUTPUT_FILENAME = "results_dev_add-one.txt"
DEV_KNESER_NEY_RESULTS_OUTPUT_FILENAME = "results_dev_kneser-ney.txt"
TES_NO_SMOOTHING_RESULTS_OUTPUT_FILENAME = "results_test_unsmoothed.txt"
TES_ADD_ONE_RESULTS_OUTPUT_FILENAME = "results_test_add-one.txt"
TES_KNESER_NEY_RESULTS_OUTPUT_FILENAME = "results_test_kneser-ney.txt"

TUNED_N_FOR_NO_SMOOTHING = 2
TUNED_N_FOR_ADD_ONE = 3
FIXED_N_FOR_KNESER_NEY = 3

KNESER_NEY_DISCOUNT = 0.75
INFINITY = float('inf')
PROB_OF_KN_ZERO_PROB_NGRAM = 0.0001 # To resolve the issue that NLTK's Kneser-Ney model may assign zero probabilities to unseen n-grams, 
									# we replace those zero probabilities with this small constant when using NLTK's Kneser-Ney smoothing model.

# Global Variables
gl_smoothing_type = 0
gl_testobj_set_root = ""
gl_testobj_file_suffix = ""

# Smoothing Type Enum
def enum(**enums):
	return type('Enum', (), enums)
	
SmoothingType = enum(NO_SMOOTHING=0, ADD_ONE=1, NLTK_KNESER_NEY=2, IMPROVED_KNESER_NEY=3)


class NgramTraModel:
	"""N-gram character model built with the given n value and training file"""
	
	def __init__(self, n, tra_filename):
	
		self.tra_filename = tra_filename
		self.n = n
				
		self.tra_letter_list = []
		self.ngram_list = []
		self.n1gram_list = []
		
		with open(TRA_SET_ROOT + tra_filename, "r") as file:
			for sentence in file:
				sentence = sentence.replace('\n','')
				# sentence_letter_list = list(nltk.pad_sequence(sentence, self.n, 
					# pad_left=False, pad_right=False))
				sentence_letter_list = list(nltk.pad_sequence(sentence, self.n, 
					pad_left=True, pad_right=True, 
					left_pad_symbol='<s>',right_pad_symbol = '</s>'))
			
				self.tra_letter_list.extend(sentence_letter_list)
				self.ngram_list.extend(list(nltk.ngrams(sentence_letter_list, self.n)))
				self.n1gram_list.extend(list(nltk.ngrams(sentence_letter_list, self.n-1)))
		
		self.V = len(set(self.tra_letter_list))
		self.ngram_cfd = nltk.FreqDist(self.ngram_list)
		self.n1gram_cfd = nltk.FreqDist(self.n1gram_list)
		
		if gl_smoothing_type == SmoothingType.NLTK_KNESER_NEY:
			self.kneser_ney_prob_dist = nltk.KneserNeyProbDist(self.ngram_cfd, bins=None, discount=KNESER_NEY_DISCOUNT)
		if gl_smoothing_type == SmoothingType.IMPROVED_KNESER_NEY:
			self.n2gram_cfd = nltk.FreqDist(self.tra_letter_list)
			
			self.letter_count_after_n1gram = {}
			self.letter_typenum_after_n1gram = {}
			self.letter_count_after_n2gram = {}
			self.letter_typenum_after_n2gram = {}
			
			for w0, w1, w2 in self.ngram_cfd:
				if (w0, w1) not in self.letter_count_after_n1gram:
					self.letter_count_after_n1gram[(w0, w1)]=0
					self.letter_typenum_after_n1gram[(w0, w1)]=0
					
				self.letter_count_after_n1gram[(w0, w1)]+=self.ngram_cfd[(w0, w1, w2)]
				self.letter_typenum_after_n1gram[(w0, w1)]+=1
				
			for w0, w1 in self.n1gram_cfd:
				if w0 not in self.letter_count_after_n2gram:
					self.letter_count_after_n2gram[w0]=0
					self.letter_typenum_after_n2gram[w0]=0
				self.letter_count_after_n2gram[w0]+=self.n1gram_cfd[(w0, w1)]
				self.letter_typenum_after_n2gram[w0]+=1
	
	def ngram_logprob(self, ngram, smoothing_type):
		"""Return the logarithm of the probability of an ngram using the given smoothing method"""
	
		if smoothing_type == SmoothingType.NO_SMOOTHING:
			ngram_count = self.ngram_cfd[ngram]
			n1gram_count = self.n1gram_cfd[ngram[:len(ngram)-1]]

			if ngram_count == 0:
				return -INFINITY
				
			if(n1gram_count == 0):
				print "Error: When ngram=",ngram,", ngram_count>0 and n1gram_count=0"
			
			return math.log(ngram_count/n1gram_count)

		elif smoothing_type == SmoothingType.ADD_ONE:
			ngram_count = self.ngram_cfd[ngram]

			n1gram_count = self.n1gram_cfd[ngram[:len(ngram)-1]] 
		
			return math.log((ngram_count+1)/(n1gram_count+self.V))
		elif smoothing_type == SmoothingType.NLTK_KNESER_NEY:
			if self.n != 3:
				print "Warning: Kneser Ney smoothing API is only applicable to trigram!"
				return
				
			prob = self.kneser_ney_prob_dist.prob(ngram)
			
			if prob != 0:
				return math.log(prob)
			else:
				return math.log(PROB_OF_KN_ZERO_PROB_NGRAM)
						
		elif smoothing_type == SmoothingType.IMPROVED_KNESER_NEY:
			if self.n != 3:
				print "Warning: Kneser Ney smoothing API is only applicable to trigram!"
				return
			
			return self.ngram_kn_improved_prob(KNESER_NEY_DISCOUNT, ngram)
	
	def ngram_sequence_logprob(self, sequence_ngram_list, smoothing_type):
		"""Return the logarithm of the probability of a sequence of ngrams using the given smoothing method"""
	
		rt = 0
		for ngram_element in sequence_ngram_list:
			logprob = self.ngram_logprob(ngram_element, smoothing_type)
			rt += logprob
			
		return rt
	
	def ngram_sequence_perlexity(self, sequence_ngram_list, smoothing_type):
		"""Return the perplexity of a sequence of ngrams using the given smoothing method"""
	
		return math.exp(
		-self.ngram_sequence_logprob(sequence_ngram_list, smoothing_type)/len(sequence_ngram_list))
	
	def ngram_kn_improved_prob(self, d, ngram):
		"""Return the probability of an ngram based on the Kneser-Ney smoothing principle introduced in our tutorial"""
	
		if self.n2gram_cfd[ngram[2]]>0:
			cont1 = 1
		else:
			cont1 = 0
		pcont1 = max(cont1 - d, 0)/self.V + d/self.V
		
		if ngram[1] not in self.letter_typenum_after_n2gram:
			lambda2 = d
			pcont2 = lambda2 * pcont1
		else: # self.letter_typenum_after_n2gram[ngram[1]]>0 && self.letter_count_after_n2gram[ngram[0:2]]>0
			
			lambda2 = d/self.letter_count_after_n2gram[ngram[1]] * self.letter_typenum_after_n2gram[ngram[1]]
		
			if self.n1gram_cfd[ngram[1:3]]>0:
				cont2 = 1
			else:
				cont2 = 0
			
			pcont2 =  max(cont2 - d, 0)/self.letter_typenum_after_n2gram[ngram[1]] + lambda2 * pcont1
		
		if ngram[0:2] not in self.letter_typenum_after_n1gram:
			lambda3 = d
			pcont3 = lambda3 * pcont2
		else: # self.letter_typenum_after_n1gram[ngram[0:2]]>0 && self.letter_count_after_n1gram[ngram[0:2]]>0

			lambda3 = d/self.letter_count_after_n1gram[ngram[0:2]] * self.letter_typenum_after_n1gram[ngram[0:2]]
					
			pcont3 = max(self.ngram_cfd[ngram] - d, 0)/self.letter_count_after_n1gram[ngram[0:2]] + lambda3 * pcont2
		
		return math.log(pcont3)

		
class NgramTestData:
	"""Data retrieved from a test file for the test of the ngram model"""
	
	def __init__(self, n, test_filename):
		self.test_filename = test_filename
		self.ngram_list = []
	
		with open(gl_testobj_set_root + test_filename, "r") as file:
			for sentence in file:
				sentence = sentence.replace('\n','')
				# sentence_letter_list = list(nltk.pad_sequence(sentence, n, 
					# pad_left=False, pad_right=False))
				sentence_letter_list = list(nltk.pad_sequence(sentence, n, 
					pad_left=True, pad_right=True, 
					left_pad_symbol='<s>',right_pad_symbol = '</s>'))
		
				self.ngram_list.extend(list(nltk.ngrams(sentence_letter_list,n)))

				
def match_files_and_print_results(test_data_list, tra_model_list, smoothing_type, need_eval = False, need_output=False, output_filename=""):
	"""Match each test file with the most likely corresponding training file by finding the training model that can get the lowest perplexity for each test file.
	This function can also print the matching result and output the result to external text files.
	"""
	n = tra_model_list[0].n
	
	if need_eval == True:
		correct_match_count = 0
		average_perplexity = 0
		
	if need_output == True:
		output_file = open(output_filename, 'w')
	
	for test_data in test_data_list:
		matched_tra_filename = "Null"
		min_perplexity = INFINITY
		for tra_model in tra_model_list:
			perplexity = tra_model.ngram_sequence_perlexity(test_data.ngram_list, smoothing_type)
		
			if(perplexity < min_perplexity):
				matched_tra_filename = tra_model.tra_filename
				min_perplexity = perplexity

		match_result = "    ".join([test_data.test_filename, matched_tra_filename, str(min_perplexity), str(n)])

		print match_result
		
		if need_output == True:
			output_file.write(match_result+'\n')
		
		if need_eval == True:
			if(test_data.test_filename[:-4] == matched_tra_filename[:-4]):
				correct_match_count += 1
				average_perplexity += min_perplexity
	
	if need_output == True:
		output_file.close()
		
	if need_eval == True:
		print ""
		print "Correct Match Rate:", correct_match_count, "/", len(test_data_list)
		
		if correct_match_count != 0:
			average_perplexity = average_perplexity/correct_match_count
		else:
			average_perplexity = "-"
		print "Matched Files Average Perplexity:", average_perplexity
		return correct_match_count, average_perplexity
	else:
		print ""
		print "Correct Match Rate: ???"
		print "Matched Files Average Perplexity: ???"
		
def main():
	"""The first function to be called when the program starts"""
	
	if len(sys.argv) != 3:
		print "Error: Wrong number of command line arguments!"
		print ""
		print "Usage: python langid.py --SMOOTHING_TYPE --TEST_OBJECT"
		print ""
		return
	
	if sys.argv[1] not in [NO_SMOOTHING_CMD, ADD_ONE_CMD, NLTK_KNESER_NEY_CMD, IMPROVED_KNESER_NEY_CMD]:
		print "Error: Invalid SMOOTHING_TYPE argument!"
		print ""
		print "Usage: python langid.py --SMOOTHING_TYPE --TEST_OBJECT"
		print ""
		print "Available --SMOOTHING_TYPE arguments:"
		print "    --unsmoothed"
		print "    --laplace"
		print "    --nltk-kneser-ney"
		print "    --improved-kneser-ney"
		print ""
		return 
		
	if sys.argv[2] not in [DEV_SET_CMD, TES_SET_CMD]:
		print "Error: Invalid TEST_OBJECT argument!"
		print ""
		print "Usage: python langid.py --SMOOTHING_TYPE --TEST_OBJECT"
		print ""
		print "Available --TEST_OBJECT arguments:"
		print "    --dev"
		print "    --tes"
		print ""
		return
		
	smoothing_type_cmd = sys.argv[1]
	test_obj_cmd = sys.argv[2]
	
	global gl_smoothing_type
	global gl_testobj_set_root
	global gl_testobj_file_suffix
	
	if test_obj_cmd == DEV_SET_CMD:
		
		gl_testobj_set_root = DEV_SET_ROOT
		gl_testobj_file_suffix = DEV_FILE_SUFFIX
		
		if smoothing_type_cmd == NO_SMOOTHING_CMD:
			gl_smoothing_type = SmoothingType.NO_SMOOTHING
			results_output_filename = DEV_NO_SMOOTHING_RESULTS_OUTPUT_FILENAME
		elif smoothing_type_cmd == ADD_ONE_CMD:
			gl_smoothing_type = SmoothingType.ADD_ONE
			results_output_filename = DEV_ADD_ONE_RESULTS_OUTPUT_FILENAME
		elif smoothing_type_cmd == NLTK_KNESER_NEY_CMD:
			gl_smoothing_type = SmoothingType.NLTK_KNESER_NEY
			results_output_filename = DEV_KNESER_NEY_RESULTS_OUTPUT_FILENAME
		elif smoothing_type_cmd == IMPROVED_KNESER_NEY_CMD:
			gl_smoothing_type = SmoothingType.IMPROVED_KNESER_NEY
			results_output_filename = DEV_KNESER_NEY_RESULTS_OUTPUT_FILENAME
			
		if smoothing_type_cmd == NLTK_KNESER_NEY_CMD or smoothing_type_cmd == IMPROVED_KNESER_NEY_CMD:
			need_tune_n = False
			min_n = FIXED_N_FOR_KNESER_NEY
			max_n = FIXED_N_FOR_KNESER_NEY
		else:
			need_tune_n = True
			min_n = int(input("To tune the parameter n, please input the min value of n: "))
			max_n = int(input("To tune the parameter n, please input the max value of n: "))
			if max_n < min_n:
				print "Error: max_n can't be smaller than min_n!"
				print ""
				return
			if min_n < 2:
				print "Error: n can't be smaller than 2!"
				print ""
				return
				
	elif test_obj_cmd == TES_SET_CMD:
		gl_testobj_set_root = TES_SET_ROOT
		gl_testobj_file_suffix = TES_FILE_SUFFIX
		
		if smoothing_type_cmd == NO_SMOOTHING_CMD:
			gl_smoothing_type = SmoothingType.NO_SMOOTHING
			results_output_filename = TES_NO_SMOOTHING_RESULTS_OUTPUT_FILENAME
		elif smoothing_type_cmd == ADD_ONE_CMD:
			gl_smoothing_type = SmoothingType.ADD_ONE
			results_output_filename = TES_ADD_ONE_RESULTS_OUTPUT_FILENAME
		elif smoothing_type_cmd == NLTK_KNESER_NEY_CMD:
			gl_smoothing_type = SmoothingType.NLTK_KNESER_NEY
			results_output_filename = TES_KNESER_NEY_RESULTS_OUTPUT_FILENAME
		elif smoothing_type_cmd == IMPROVED_KNESER_NEY_CMD:
			gl_smoothing_type = SmoothingType.IMPROVED_KNESER_NEY
			results_output_filename = TES_KNESER_NEY_RESULTS_OUTPUT_FILENAME
		need_tune_n = False
		
		if smoothing_type_cmd == NO_SMOOTHING_CMD:
			min_n = TUNED_N_FOR_NO_SMOOTHING
			max_n = TUNED_N_FOR_NO_SMOOTHING
		elif smoothing_type_cmd == ADD_ONE_CMD:
			min_n = TUNED_N_FOR_ADD_ONE
			max_n = TUNED_N_FOR_ADD_ONE
		elif smoothing_type_cmd == NLTK_KNESER_NEY_CMD or smoothing_type_cmd == IMPROVED_KNESER_NEY_CMD:
			min_n = FIXED_N_FOR_KNESER_NEY
			max_n = FIXED_N_FOR_KNESER_NEY
	
	if need_tune_n == True:
		best_n = 0
		max_match_count = 0
		best_match_average_perplexity = INFINITY
		best_ngram_model_list = []
		best_ngram_test_data_list = []
		
	for n in range(min_n, max_n+1):

		ngram_tra_model_list = []
		
		for filename in os.listdir(TRA_SET_ROOT):
			if filename.endswith(TRA_FILE_SUFFIX):
				tra_filename = filename
				ngram_tra_model_list.append(NgramTraModel(n, tra_filename))
		
		ngram_test_data_list = []
		
		for filename in os.listdir(gl_testobj_set_root):
			if filename.endswith(gl_testobj_file_suffix):
				test_filename = filename
				ngram_test_data_list.append(NgramTestData(n, test_filename))
		
		print ""
		print "="*10, "Match Files When n Equals", n, "="*10
		evals = match_files_and_print_results(ngram_test_data_list, ngram_tra_model_list, gl_smoothing_type, 
			test_obj_cmd == DEV_SET_CMD, 
			False, results_output_filename)
		
		if need_tune_n == True:
			match_count = evals[0]
			average_perplexity = evals[1]
			
			if (match_count>max_match_count) or \
			(match_count==max_match_count and average_perplexity<best_match_average_perplexity):
				best_n = n
				max_match_count = match_count
				best_match_average_perplexity = average_perplexity
				best_ngram_model_list = ngram_tra_model_list
				best_ngram_test_data_list = ngram_test_data_list
				
	if need_tune_n == True:
		print ""
		print "="*10, "The Result of Tunning Parameter n", "="*10
		if best_n != 0:
			print "The Tuned Parameter n:", best_n
			print "Max Match Rate:", max_match_count,"/",len(best_ngram_test_data_list)
			print "Best Match Average Perplexity:", best_match_average_perplexity
			
			print ""
			print "="*10, "Matched Files With The Tuned n Value", "="*10
			match_files_and_print_results(best_ngram_test_data_list, best_ngram_model_list, gl_smoothing_type, True, False, results_output_filename)
		else:
			print "No n value is suitable."
			
	print ""
	
if __name__ == '__main__':
	main()
	os.system("pause")