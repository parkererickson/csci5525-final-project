import pyTigerGraph as tg
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sklearn
import optuna
import wandb
import matplotlib.pyplot as plt
import pickle

conn = tg.TigerGraphConnection(graphname="SCOTUS_Graph")

edgeTypes = conn.getEdgeTypes() + ["STATE_CONTAINS_PETITIONER", 
                                   "STATE_CONTAINS_RESPONDENT", "PETITIONER_TYPE_IN_CASE", 
                                   "RESPONDENT_TYPE_IN_CASE", "ORIGIN_OF_CASE",
                                   "STATE_ORIGIN_OF_CASE", "ISSUE_OF_CASE",
                                   "AREA_CONTAINS_ISSUE", "VOTED_CONSERVATIVE",
                                   "VOTED_LIBERAL"]  # Have to get reverse edges too, except CITED_BY, which would be forward-looking

vertexTypes = conn.getVertexTypes()
with open("../graph_creation/gsql/queries/tg_fastRP.gsql", "r") as f:
    fast_rp_query = f.read()

labels = {"LIBERAL_VOTE": 1,
          "CONSERVATIVE_VOTE": 0}

justices = conn.getVertices("Justice")
justices = [x["v_id"] for x in justices]

def createData(justices, embeddingDim, split):
    Xdata = []
    ydata = []
    for justice in justices:
        res = conn.runInstalledQuery("justiceCaseLinks", params={"justiceID": int(justice), split:True})
        caseEmbeddings = pd.DataFrame.from_dict(res[0]["@@caseEmbeddings"], orient="index").reset_index().rename(columns={"index":"caseId"})
        caseVotes = pd.DataFrame.from_dict(res[0]["@@caseVote"], orient="index").reset_index().rename(columns={"index":"caseId", 0:"vote"})
        data = caseVotes.merge(caseEmbeddings, on="caseId")
        justiceEmbedding = pd.DataFrame.from_dict(res[0]["@@justiceEmbedding"], orient="index").reset_index().rename(columns={"index":"justice"})
        X = data[[i for i in range(0,embeddingDim)]].values
        je = justiceEmbedding[[i for i in range(0,embeddingDim)]].values
        je = np.tile(je, (X.shape[0], 1))
        X = np.append(X, je, axis=1)
        y = np.array([labels[x] for x in data["vote"]])
        Xdata.append(X)
        ydata.append(y)
    return np.concatenate(Xdata), np.concatenate(ydata)

def createTransform(X, embedding_dim=200, method="L2"):
    case = X[:,:embedding_dim]
    justice = X[:,embedding_dim:]
    if method == "L2":
        return (case-justice)**2
    elif method == "hadamard":
        return np.multiply(case, justice)
    elif method == "L1":
        return np.abs(case-justice)
    elif method == "avg":
        return (case+justice)/2
    elif method == "concat":
        return X
    else:
        raise("Invalid transform method")

def constructParamUrl(v_types, e_types, params):
    paramUrl = ""
    for vType in v_types:
        paramUrl += ("v_type="+vType+"&")
    for eType in e_types:
        paramUrl += ("e_type="+eType+"&")
    for p in params.keys():
        paramUrl += (p+"="+str(params[p])+"&")
    paramUrl = paramUrl[:-1]
    return paramUrl

params = {
    "weights":"4,2,1",
    "beta":-0.5,
    "k":3,
    "reduced_dim":200, 
    "sampling_constant":3,
    "random_seed":42, 
    "result_attr":"embedding"
}


def objective(trial):
    #params["beta"] = trial.suggest_float("beta", -1,1)
    params["beta"] = trial.suggest_float("beta", -1,0.1)
    #params["reduced_dim"] = trial.suggest_categorical("reduced_dim", [64, 128, 256, 512])
    params["reduced_dim"] = trial.suggest_categorical("reduced_dim", [256, 512])
    #params["sampling_constant"] = trial.suggest_int("sampling_constant", 1,5)
    params["sampling_constant"] = trial.suggest_int("sampling_constant", 1,3)
    #params["weights"] = trial.suggest_categorical("weights", ["1,1,1", "1,2,1", "1,4,1", "2,2,1", "2,4,1", "3,4,1", "1,2,4", "1,3,4", "1,4,4", "2,3,4", "4,2,1"])
    params["weights"] = trial.suggest_categorical("weights", ["1,1,1", "1,3,4", "1,4,4", "2,3,4"])
    #transform = trial.suggest_categorical("transform", ["L2", "hadamard", "L1", "avg", "concat"])
    transform = trial.suggest_categorical("transform", ["concat"])
    params["transform"] = transform
    paramUrl = constructParamUrl(vertexTypes, edgeTypes, params)
    conn.gsql("USE GRAPH "+conn.graphname+"\n"+"DROP QUERY tg_fastRP")
    queryDef = "USE GRAPH "+conn.graphname+"\n"+fast_rp_query.replace("@embedding_dim@", str(params["reduced_dim"]))
    conn.gsql(queryDef)
    wandb.init(project="scotus-citation-graph", config=params)
    conn.runInstalledQuery("tg_fastRP", params=paramUrl, timeout=512_000)
    
    X_train, y_train = createData(justices, params["reduced_dim"], split="train")
    X_train = createTransform(X_train, params["reduced_dim"], transform)
    X_val, y_val = createData(justices, params["reduced_dim"], split="valid")
    X_val = createTransform(X_val, params["reduced_dim"], transform)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_val, y_val)
    
    model_name = "fastRP_"+transform+"_"+str(params["reduced_dim"])+"_"+str(params["sampling_constant"])+"_"+str(params["weights"])+"_"+str(params["beta"])
    wandb.sklearn.plot_classifier(clf, X_train, X_val, y_train, y_val, clf.predict(X_val), clf.predict_proba(X_val), labels=["Conservative Vote","Liberal Vote"], model_name=model_name)

    pca = sklearn.decomposition.PCA(n_components=3)
    X = np.concatenate([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    X_red = pca.fit_transform(X)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_red[:,0], X_red[:,1], X_red[:,2], c=y)
    ax.legend(labels=["Conservative Vote","Liberal Vote"])
    wandb.log({"pca" : plt,
               "pca_explained_var": pca.explained_variance_ratio_})
    wandb.log({"accuracy": accuracy})
    wandb.finish()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print(study.best_trial)
    with open("citation_edges_study_2.pkl", "wb") as f:
        pickle.dump(study, f)
    
