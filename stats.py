## Utility functions for statistically analyzing batch results
## accuracy, error, and other relevant metrics

## Import relevant libraries from main file
import main

## Importing sklearn metrics
from sklearn.metrics import mean_squared_error 

## Returns MSE, F1 score, precision, and recall from batch results
def stats_analysis (batchResults):
  ## Convert batch results to dataframe
  batchResultsDF = main.pd.DataFrame(batchResults)
  correct = batchResultsDF[batchResultsDF['correct'] == 1]
  incorrect = batchResultsDF[batchResultsDF['correct'] == 0]

  ## Calculate MSE
  mse = mean_squared_error(batchResultsDF['GPTtoxicity'], batchResultsDF['toxicityHuman'])
  print('MSE:', mse)

  ## Calculate F1 score
  true_positive = correct[correct['GPTtoxicity'] >= 3].shape[0]
  ## true_negative = correct[correct['GPTtoxicity'] < 3].shape[0]
  false_positive = incorrect[incorrect['GPTtoxicity'] >= 3].shape[0]
  false_negative = incorrect[incorrect['GPTtoxicity'] < 3].shape[0]
  
  ## Precision and recall
  precision = true_positive / (true_positive + false_positive)
  recall = true_positive / (true_positive + false_negative)
  f1 = 2 * (precision * recall) / (precision + recall)

  ## Output results
  print(f'F1 score: {f1}\nPrecision: {precision}\nRecall: {recall}')
  return mse, f1, precision, recall
