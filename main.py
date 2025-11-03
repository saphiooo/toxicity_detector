## Main file for Toxic Content Detector

#### SETTING UP ####

## Importing relevant libraries
from openai import OpenAI
import pandas as pd
import torch
import pandas as pd

## Importing relevant utility functions
from stats import stats_analysis
from batch import evaluateBatchText, parseBatchOutput

## Reading in ToxiGen data, splitting into testing and training sets
splits = {'test': 'annotated/test-00000-of-00001.parquet', 'train': 'annotated/train-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/toxigen/toxigen-data/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/toxigen/toxigen-data/" + splits["test"])

## Pre-loading in RAG database
db = pd.read_csv('database.csv')
database = []
for row in db.itertuples():
  s = f'Sentence: {row.sentence} \nToxicity level: {row.toxicityHuman}.\nRationale: {row.rationaleHuman}'
  database.append(s)

## OpenAI API key and client setup
## NOTE: Paste your API key below
api_key = 'API_KEY_HERE'
client = OpenAI(api_key=api_key)
## Context
CONTEXT = '''You are a toxic content detector, detecting hate speech, discrimination, and violence directed harmfully toward a person or group of people.'''


#### CONSTANTS ####
DATABASE_SIZE = len(database)
DIM = 1024
TOPK = 4


#### MAIN #### 
## NOTE: If no database embeddings file exists, uncomment the
## following line to create one; this only needs to be done once. 
## Additionally, comment out the line that loads pre-made embeddings.
# make_embeddings(database)

## Load pre-made database embeddings
database_embeddings = torch.load('database_embeddings.pt')

## Run batch evaluations with RAG on test set
total = 0
batchResults = {'sentence': [], 'GPTtoxicity': [], 'toxicityHuman': [], 'rationale': [], 'correct': []}
query = []
truth = []
output = []
BATCH_SIZE = 10
TEST_SIZE = 100
average_time = 0
ct = 0

for row in df_test[:TEST_SIZE].itertuples():
  total += 1
  print(row)
  if (len(query) < BATCH_SIZE):
    query.append(row.text)
    truth.append(row.toxicity_human)
  else:
    [response, response_time] = evaluateBatchText(query, BATCH_SIZE)
    average_time += response_time
    ct += 1
    parsed = parseBatchOutput(response, BATCH_SIZE)
    for i in range(len(parsed)):
      batchResults['sentence'].append(query[i])
      batchResults['GPTtoxicity'].append(parsed[i][2])
      batchResults['toxicityHuman'].append(truth[i])
      batchResults['rationale'].append(parsed[i][3])

      if ((parsed[i][2] < 3 and truth[i] < 3) or
        (parsed[i][2] >= 3 and truth[i] >= 3)):
        batchResults['correct'].append(1)
      else:
        batchResults['correct'].append(0)
    query = [row.text]
    truth = [row.toxicity_human]
[response, response_time] = evaluateBatchText(query, BATCH_SIZE)
average_time += response_time
ct += 1
parsed = parseBatchOutput(response, BATCH_SIZE)
for i in range(len(parsed)):
  batchResults['sentence'].append(query[i])
  batchResults['GPTtoxicity'].append(parsed[i][2])
  batchResults['toxicityHuman'].append(truth[i])
  batchResults['rationale'].append(parsed[i][3])

  if ((parsed[i][2] < 3 and truth[i] < 3) or
      (parsed[i][2] >= 3 and truth[i] >= 3)):
    batchResults['correct'].append(1)
  else:
    batchResults['correct'].append(0)

print('Raw accuracy:', sum(batchResults['correct'])/total)
print('Average time:', average_time/ct)

## Analyzing results
stats_analysis(batchResults)
