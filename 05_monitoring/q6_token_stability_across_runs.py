import sqlite3
import pandas as pd

conn = sqlite3.connect("traces.db")
df = pd.read_sql("SELECT * FROM spans", conn)

llm_spans = df[df.name == "llm"].reset_index(drop=True)
print(llm_spans[["input_tokens", "output_tokens", "cost"]])

variation = (
    (llm_spans.input_tokens.max() - llm_spans.input_tokens.min())
    / llm_spans.input_tokens.mean()
)
print(f"input_tokens range as % of mean: {variation:.2%}")