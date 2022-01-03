

## MultiPlexDataset
import torch
from typing import List
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, T5ForConditionalGeneration
import numpy as np
##

## errorCalculator
import nltk
import nltk.translate.gleu_score as gleu
import nltk.translate.bleu_score as bleu
from tqdm.auto import tqdm 

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')
##




class MultiLexDataset(Dataset):

    def __init__(self,
                 path_to_files: List[str],
                 only_include_corrections: bool = False,
                 short_data: bool = False):
        """

        :param path_to_files: List of paths to the files with data
        :param only_include_corrections: Whether to only include samples where there are corrections
        """

        self.only_include_corrections = only_include_corrections
        self.dataset_counter = 0
        self.data = {}

        self.test = {}
        self.train = {}
        self.validation = {}
        self.originalTest = {}
        self.original_sentences = {}
        self.original_sentences_counter = 0

        print("Loading data...")
        for path in path_to_files:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read().split("\n")

            current_norm_words = []
            current_ref_words = []

            for line_num,line in enumerate(data):
              if short_data and line_num > 50000: 
                break 
              if not line:
                  #self.create_samples(current_norm_words, current_ref_words)
                  
                  self.original_sentences[self.original_sentences_counter] = {"Transcription": current_norm_words, "Reference": current_ref_words}
                  self.original_sentences_counter += 1
                  
                  current_ref_words = []
                  current_norm_words = []
              else:
                  norm, ref = line.split("\t")
                  current_norm_words.append(norm)
                  current_ref_words.append(ref)
        self.originalDataset = data
        print("Dataset initialized...")

        # Split into train (70%), validation (15%) and test (15%) set
        trainSplitSize = round(len(self.original_sentences)*0.995)
        tempTrain, tempValidationTest = torch.utils.data.random_split(self.original_sentences, [trainSplitSize, self.original_sentences.__len__()-trainSplitSize])
        tempValidation, tempTest = torch.utils.data.random_split(tempValidationTest, [int(np.floor(len(tempValidationTest)*0.5)), int(np.ceil(len(tempValidationTest)*0.5))])

        self.originalTest = tempTest

        # Call create_sample to insert ids
        self.data = {}
        for i in range(len(tempTrain)):
            
            self.create_samples(tempTrain[i]["Transcription"], tempTrain[i]["Reference"],testbool=False)
        self.train = self.data

        self.dataset_counter = 0
        self.data = {}
        for i in range(len(tempValidation)):
            
            self.create_samples(tempValidation[i]["Transcription"], tempValidation[i]["Reference"],testbool=False)
        self.validation = self.data

        self.dataset_counter = 0
        self.data = {}
        for i in range(len(tempTest)):
            
            self.create_samples(tempTest[i]["Transcription"], tempTest[i]["Reference"], testbool=True)
        self.test = self.data

        sumLen = len(tempTrain) + len(tempValidation) + len(tempTest)

        print("Training data:\t\t" + str(len(tempTrain)) + '\t ' + str(round(100*len(tempTrain)/sumLen)) + '%')

        print("Validation data:\t" + str(len(tempValidation)) + '\t ' + str(round(100*len(tempValidation)/(sumLen))) + '%')

        print("Test data:\t\t" + str(len(tempTest)) + '\t ' + str(round(100*len(tempTest)/(sumLen))) + '%')

        

    def create_samples(self, norms, refs, testbool):
        if norms and refs:
            for i, word in enumerate(norms):

                if self.only_include_corrections and word == refs[i] and not(testbool):
                    continue

                if i == 0:
                    sample_input = "<extra_id_0>" + word + "<extra_id_1> " + " ".join(norms[i + 1:])
                elif i == len(norms) - 1:
                    sample_input = " ".join(norms[:i]) + " <extra_id_0>" + word + "<extra_id_1>"
                else:
                    sample_input = " ".join(norms[:i]) + " <extra_id_0>" + word + "<extra_id_1> " + " ".join(
                        norms[i + 1:])

                self.data[self.dataset_counter] = {"input_sample": sample_input, "expected_output": refs[i]}
                self.dataset_counter += 1

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data.keys())

class CollateFunctor:
    def __init__(self, tokenizer, encoder_max_length=320, decoder_max_length=32):
        self.tokenizer = tokenizer
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length

    def __call__(self, samples):
        inputs = list(map(lambda x: x["input_sample"], samples))

        inputs = self.tokenizer(
            inputs, padding=True, truncation=True, pad_to_multiple_of=8,
            max_length=self.encoder_max_length, return_attention_mask=True, return_tensors='pt'
        )

        outputs = list(map(lambda x: x["expected_output"], samples))

        outputs = self.tokenizer(
            outputs, padding=True, truncation=True, pad_to_multiple_of=8,
            max_length=self.decoder_max_length, return_attention_mask=True, return_tensors='pt'
        )

        batch = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": outputs.input_ids,
            "decoder_attention_mask": outputs.attention_mask
        }
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100  # used to mask the loss in T5
        return batch



class errorCalculator:

  def __init__(self,sentenceList):

    self.errorMetrics = {}
    self.errorMeanMetrics = {"wer": 0, "bleu": 0, "gleu": 0}
    #progress_bar = tqdm(range(len(sentenceList)))
    
    # Loop over wer_score, 
    for i,sentence in enumerate(sentenceList):
      current_transcription = sentenceList[i]["Transcription"]
      current_reference = sentenceList[i]["Reference"]
      
      werScore = self.wer(current_reference, current_transcription)
      bleuScore = bleu.sentence_bleu([current_reference],current_transcription)
      gleuScore = gleu.sentence_gleu([current_reference], current_transcription)

      if(not(np.isnan(werScore)) and not(np.isnan(bleuScore)) and not(np.isnan(gleuScore))):
          self.errorMeanMetrics = {"wer": self.errorMeanMetrics["wer"] + werScore, "bleu": self.errorMeanMetrics["bleu"] + bleuScore, "gleu": self.errorMeanMetrics["gleu"] + gleuScore}

      self.errorMetrics[i] = {"wer": werScore, "bleu": bleuScore, "gleu": gleuScore}

      #progress_bar.update(1)
      



    self.errorMeanMetrics = {"wer": self.errorMeanMetrics["wer"]/len(sentenceList), "bleu": self.errorMeanMetrics["bleu"]/len(sentenceList), "gleu": self.errorMeanMetrics["gleu"]/len(sentenceList)}

  def wer(self, r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return 1 - d[len(r)][len(h)]/len(h)




class testsetAgainstNLPMetrics:

    def __init__(self, dataset, tokenizer, model, device):
        i = 0
        self.collection = {}
        
        while True: #range(len(dataset.test)):
            if i >= len(dataset.test): break

            sentenceLength = len(dataset.test[i]["input_sample"].split())
            
            #print(sentenceLength)  

            # For words in the sentence
            currentSentenceList = []
            for j in range(sentenceLength):

                currentSentenceList.append(dataset.test[i+j]["input_sample"])

            #print(currentSentenceList)
             
            tokenizedSentences = tokenizer(currentSentenceList, padding=True, truncation=False, pad_to_multiple_of=8,
                                return_attention_mask=True, return_tensors='pt')

            tokenizedSentences.to(device)

            output = model.generate(
                input_ids=tokenizedSentences['input_ids'],
                attention_mask=tokenizedSentences['attention_mask'],
                )


            del tokenizedSentences

            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
            #print(dataset.originalTest[len(self.collection)]["Reference"])
            #print(decoded_output)
            #print("\n")
            self.collection[len(self.collection)] = {"Transcription": decoded_output, "Reference": dataset.originalTest[len(self.collection)]["Reference"], "Input": currentSentenceList[0]}

            i = i + sentenceLength
            
        
        self.errorCalc = errorCalculator(self.collection)
        #print(errorCalc.errorMeanMetrics)

def modelFreezeStatus(model):
	
	for name, param in model.named_parameters():
		print(name)
		print(param.requires_grad)