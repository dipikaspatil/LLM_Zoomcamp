from embedder import Embedder

embed = Embedder()
query = "How does approximate nearest neighbor search work?"
v = embed.encode(query)

print(v.shape)   # (384,)
print(v)
print(v[0])      # first component of the vector