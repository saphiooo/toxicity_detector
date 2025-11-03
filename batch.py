## batch.py
## Driver function of batch processing

## Import relevant libraries
import re 
import time

## Import relevant functions and variables from main file
import main
from vector import retrieve_examples, get_embeddings

## Evaluates batch of text inputs, returning GPT toxicity evaluations
def evaluateBatchText (text, n):
  ## Load in relevant examples from database
  examples = []
  input_embeddings = main.torch.Tensor([get_embeddings(text[i]) for i in range(len(text))])
  _, _, _, examples = retrieve_examples(input_embeddings, main.database_embeddings, main.database, k=main.TOPK)

  ## Load input sentences into singular prompt
  questions = ''
  for i in range(n):
    questions += str(i + 1) + '. ' + text[i] + '\n'
  ex_sentences = ''
  for i in range(n):
    ex_sentences += str(i + 1) + '.\n' + examples[i] + '\n'
  question = f'''Analyze the following {n} sentences for toxic content:
{questions}
Let's think through this step by step. For each sentence, answer the following three questions and say nothing else:
First, is the text toxic? Answer yes or no.
Second, rate the toxicity level from 1 to 5, 1 being not toxic and 5 being very toxic.
Third, provide a brief rationale why the text is toxic or not.
Respond in this format to each sentence:
1.
Sentence: The sentence goes here.
Toxicity: Yes.
Toxicity level: 4.
Rationale: The rationale for the toxicity level goes here.
Here are relevant example responses to similar prompts:
{ex_sentences}
'''
  ## Generate response from GPT model
  input = [
      { 'role': 'developer', 'content': main.CONTEXT },
      { 'role': 'user', 'content': question },
  ]
  time_start = time.time()
  response = main.client.responses.create(
      model='gpt-4.1-nano',
      input=input
  )
  time_end = time.time()
  response_time = round(time_end - time_start, 4)

  return [response.output_text, response_time]

## Parses batch output, returning an array of [prompt number, isToxic, toxicityLevel, rationale]
def parseBatchOutput (text, n):
  ans = text.split('\n')
  res = []
  i = 0
  q = 1

  ## Remove formatting
  while ('' in ans):
    ans.remove('')

  while (q <= n):
    r = []
    ## Isolate response to one prompt
    while (i < len(ans) and ans[i][0] != str(q + 1) and ans[i][0:2] != str(q + 1)):
      r += [ans[i]]
      i += 1
    i += 1
    ## Parse individual response
    isToxic = False
    toxicityLevel = 0
    rationale = ''
    for l in r:
      try:
        if (re.search('toxicity:', l.lower()) != None):
          isToxic = re.search('.*yes.*', l.lower()) != None
        elif (re.search('toxicity level:', l.lower()) != None):
          toxicityLevel = int(re.findall('[12345]', l)[0])
        elif (re.search('rationale:', l.lower()) != None):
          rationale = l
      except:
        print('Error! Could not parse line:', l)
        return []
      
    ## Save result, increment to next response
    res.append([q, isToxic, toxicityLevel, rationale])
    q += 1
  return res
