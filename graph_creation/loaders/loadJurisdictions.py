import os
import json

def load(conn, file1="../data/court_data/", **kwargs):
    courts = []
    jurisdictions = []
    for filename in os.listdir(file1):
        if filename.endswith(".json") and filename != "info.json":
            with open(os.path.join(file1, filename)) as fp:
                courtData = json.load(fp)
                courts.append((courtData["id"], {"full_name":courtData["full_name"]}))
                jurisdictions.append((courtData["id"], courtData["jurisdiction"]))
        else:
            continue

    courtsUpserted = conn.upsertVertices("Court", courts)
    print("Upserted:", courtsUpserted, "Courts")

    jurisUpserted = conn.upsertEdges("Court", "COURT_HAS_JURISDICTION", "Jurisdiction", jurisdictions)
    print("Upserted:", jurisUpserted, "Jurisdiction/Court Edges")
