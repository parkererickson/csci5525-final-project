import pandas as pd

def load(conn, file1="../citation-graph/SCDB_2021_01_justiceCentered_Citation_cleaned.csv", **kwargs):
    df = pd.read_csv(file1)
    df["partyWinning"] = df["partyWinning"].apply(lambda x: "plantiff" if x == 1 else "defendant")
    sc_verts = df[["caseId", "caseName", "partyWinning"]].drop_duplicates()
    print(sc_verts.head())
    casesUp = conn.upsertVertexDataFrame(sc_verts, "SCCase", v_id="caseId", attributes={"caseName": "caseName", "winningParty": "partyWinning"})
    print("SCDB Cases: ", casesUp)
