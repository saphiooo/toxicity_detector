## Utility functions relating to vector embeddings and retrieval-augmented generation

## Import API key and client setup from main file
import main

## Generate vector embeddings for the RAG database and save
def make_embeddings (database):
  database_embeddings = main.torch.Tensor([
      get_embeddings(database[i]) for i in range(len(database))
  ])
  main.torch.save(database_embeddings, 'database_embeddings.pt')

## Returns vector embedding for a given sentence
def get_embeddings(sentence):
  response = main.client.embeddings.create(
      input=sentence,
      model="text-embedding-3-small"
  )
  return response.data[0].embedding

## Retrieves top K similar examples from database based on vector embeddings
def retrieve_examples (a, b, database, k=main.TOPK):
  ## Determine most similar examples in database using cosine similarity
  a = a/main.torch.norm(a, p=2, dim=-1, keepdim=True)
  b = b/main.torch.norm(b, p=2, dim=-1, keepdim=True)
  dotp = main.torch.matmul(a, b.t())
  topk_vals, topk_idxs = main.torch.topk(dotp, k=k, dim=1, largest=True, sorted=True)
  
  ## Process examples
  examples = []
  for i in topk_idxs:
    for j in i:
      examples.append(database[j])
  return dotp, topk_vals, topk_idxs, examples
