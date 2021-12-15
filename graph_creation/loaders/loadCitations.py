import pandas as pd

def load(conn, file1="../citation-graph/scdb_citations.csv", **kwargs):
    # read in data in chunks
    chunksize = 50_000
    citationUp = 0
    for chunk in pd.read_csv(file1, chunksize=chunksize, header=0, low_memory=False):
        citationUp += conn.upsertEdgeDataFrame(chunk, "SCCase", "CASE_CITED", "SCCase", from_id="citing_opinion_id", to_id="cited_opinion_id", attributes={})
    print("Citations Upserted:", citationUp)
    