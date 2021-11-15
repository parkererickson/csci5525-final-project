import pandas as pd

def load(conn, file1="../data/citations/all.csv", **kwargs):
    edges_up = 0
    for chunk in pd.read_csv(file1, chunksize=100_000):
        chunk = chunk.astype(str)
        chunk["confidence"] = 1
        edges_up += conn.upsertEdgeDataFrame(chunk, "Opinion", "OPINION_CITES", "Opinion", "citing_opinion_id", "cited_opinion_id", {"confidence": "confidence"})
    print("Upserted:", edges_up, "Edges")