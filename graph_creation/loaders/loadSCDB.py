import pandas as pd

def load(conn, file1="../citation-graph/SCDB_2021_01_justiceCentered_Citation.csv", **kwargs):
    df = pd.read_csv(file1, dtype=str, encoding = "ISO-8859-1")
    df["partyWinning"] = df["partyWinning"].apply(lambda x: "plaintiff" if x == "1" else "defendant")
    df['dateDecision'] =  pd.to_datetime(df['dateDecision'], format='%m/%d/%Y').dt.strftime("%Y/%m/%d")
    sc_verts = df[["caseId", "caseName", "partyWinning", "dateDecision"]].drop_duplicates()
    casesUp = conn.upsertVertexDataFrame(sc_verts, "SCCase", v_id="caseId", attributes={"caseName": "caseName", "winningParty": "partyWinning", "dateDecision": "dateDecision"})
    print("SCDB Cases: ", casesUp)
    areas = df[["issueArea"]].drop_duplicates()
    areasUp = conn.upsertVertexDataFrame(areas, "IssueArea", attributes={})
    print("Issue Areas:", areasUp)
    issues = df[["issue"]].drop_duplicates()
    issueUp = conn.upsertVertexDataFrame(issues, "Issue", attributes={})
    print("Issues:", issueUp)
    issueAreas = df[["issue", "issueArea"]].drop_duplicates()
    issueAreasUp = conn.upsertEdgeDataFrame(issueAreas, "Issue", "ISSUE_IN_AREA", "IssueArea", from_id="issue", to_id="issueArea", attributes={})
    print("Issue-Area Edges Up:", issueAreasUp)
    caseIssues = df[["issue", "caseId"]].drop_duplicates()
    caseIssueUp = conn.upsertEdgeDataFrame(caseIssues, "SCCase", "CASE_ISSUE", "Issue", from_id="caseId", to_id="issue", attributes={})
    print("Case-Issue Edges:", caseIssueUp)
    petitioners = df[["petitioner"]].drop_duplicates()
    petUp = conn.upsertVertexDataFrame(petitioners, "PetitionerType", attributes={})
    print("Petitioner Types:", petUp)
    petitionerCase = df[["caseId", "petitioner"]].drop_duplicates()
    petCaseUp = conn.upsertEdgeDataFrame(petitionerCase, "SCCase", "CASE_HAS_PETITIONER_TYPE", "PetitionerType", from_id="caseId", to_id="petitioner", attributes={})
    print("Case-Petitioner Type Edges:", petCaseUp)
    respondent = df[["respondent"]].drop_duplicates()
    resUp = conn.upsertVertexDataFrame(respondent, "RespondentType", attributes={})
    print("Respondent Types:", resUp)
    respondentCase = df[["caseId", "respondent"]]
    resCaseUp = conn.upsertEdgeDataFrame(respondentCase, "SCCase", "CASE_HAS_RESPONDENT_TYPE", "RespondentType", from_id="caseId", to_id="respondent", attributes={})
    print("Case-Respondent Edges:", resCaseUp)
    origins = df[["caseOrigin"]].drop_duplicates()
    originsUp = conn.upsertVertexDataFrame(origins, "OriginOfCaseType", attributes={})
    print("Origins:", originsUp)
    caseOrigins = df[["caseId", "caseOrigin"]].drop_duplicates()
    caseOriginUp = conn.upsertEdgeDataFrame(caseOrigins, "SCCase", "CASE_ORIGIN", "OriginOfCaseType", from_id="caseId", to_id="caseOrigin", attributes={})
    print("Case-Origin Edges:", caseOriginUp)
    petState = df[["petitionerState"]].drop_duplicates()
    petStateUp = conn.upsertVertexDataFrame(petState, "State", attributes={})
    print("Petitioner States:", petStateUp)
    resState = df[["respondentState"]].drop_duplicates()
    resStateUp = conn.upsertVertexDataFrame(resState, "State", attributes={})
    print("Respondent State:", resStateUp)
    originState = df[["caseOriginState"]].drop_duplicates()
    originStateUp = conn.upsertVertexDataFrame(originState, "State", attributes={})
    print("Origin States:", originStateUp)
    petStateCase = df[["caseId", "petitionerState"]].drop_duplicates()
    petStateCaseUp = conn.upsertEdgeDataFrame(petStateCase, "SCCase", "PETITIONER_IN_STATE", "State", from_id="caseId", to_id="petitionerState", attributes={})
    print("Case-Petitioner State Edges:", petStateCaseUp)
    resStateCase = df[["caseId", "respondentState"]].drop_duplicates()
    resStateCaseUp = conn.upsertEdgeDataFrame(resStateCase, "SCCase", "RESPONDENT_IN_STATE", "State", from_id="caseId", to_id="respondentState", attributes={})
    print("Case-Respondent State Edges:", resStateCaseUp)
    originStateCase = df[["caseId", "caseOriginState"]].drop_duplicates()
    oSCUp = conn.upsertEdgeDataFrame(originStateCase, "SCCase", "CASE_ORIGIN_FROM_STATE", "State", from_id="caseId", to_id="caseOriginState", attributes={})
    print("Case-Orgin State Edges:", oSCUp)

    justices = df[["justice", "justiceName"]].drop_duplicates()
    justicesUp = conn.upsertVertexDataFrame(justices, "Justice", v_id="justice", attributes={"justiceName":"justiceName"})
    print("Justices:", justicesUp)
    df["vote"] = df.apply(lambda x: "plaintiff" if x["partyWinning"] == "plaintiff" else "defendant", axis=1)
    inFavorOfPlaintiff = df[df["vote"]=="plaintiff"]
    inFavorOfPlaintiff = inFavorOfPlaintiff[["justice", "caseId"]].drop_duplicates()
    plaintiffEdgeUp = conn.upsertEdgeDataFrame(inFavorOfPlaintiff, "Justice", "VOTED_IN_FAVOR_OF_PLAINTIFF", "SCCase", from_id="justice", to_id="caseId", attributes={})
    print("Justice Votes For Plaintiff:", plaintiffEdgeUp)
    inFavorOfDefendant = df[df["vote"]=="defendant"]
    inFavorOfDefendant = inFavorOfDefendant[["justice", "caseId"]].drop_duplicates()
    defendantEdgesUp = conn.upsertEdgeDataFrame(inFavorOfDefendant, "Justice", "VOTED_IN_FAVOR_OF_DEFENDANT", "SCCase", from_id="justice", to_id="caseId", attributes={})
    print("Justice Votes For Defendant:", defendantEdgesUp)

