import pandas as pd
import numpy as np

clr_data = pd.read_csv("clr_data.csv", index_col=0)
meta = pd.read_csv("metadata.csv")
tax = pd.read_csv("final.csv")
with open("final_genus_list.txt") as f: genus_list = [x.strip() for x in f.readlines()]
meta.index = meta["SampleID"].astype(str)
common_samples = clr_data.index.intersection(meta.index)
clr_data = clr_data.loc[common_samples]
meta = meta.loc[common_samples]
tax["Genus"] = tax["Taxon"].str.extract(r"g__([^;]+)")
tax = tax[tax["Genus"].isin(genus_list)]
common_features = clr_data.columns.intersection(tax["Feature ID"])
clr_sub = clr_data[common_features]
genus_df = pd.DataFrame(index=clr_sub.index,columns=genus_list,data=0.0)
for g in genus_list:
    asvs = tax.loc[tax["Genus"] == g, "Feature ID"]
    asvs = list(set(asvs).intersection(clr_sub.columns))
    if len(asvs) > 0:
        genus_df[g] = clr_sub[asvs].sum(axis=1)
from sklearn.model_selection import train_test_split
X = genus_df
y = meta["Group"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=123,stratify=y)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=10,shuffle=True, random_state=123)

#Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf_pipe = Pipeline([("smote", SMOTE(random_state=123)),("model", RandomForestClassifier(random_state=123))])
rf_grid = {"model__n_estimators": [200, 500, 1000], "model__max_depth": [None, 3, 5, 10], "model__max_features": ["sqrt", 3, 4, 6]}
rf_model = GridSearchCV(rf_pipe, rf_grid, scoring="roc_auc", cv=cv,n_jobs=1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
print("RF Best Parameters:", rf_model.best_params_)

#SVM

from sklearn.svm import SVC
svm_pipe = Pipeline([("smote", SMOTE(random_state=123)), ("model", SVC(probability=True))])
svm_grid = {"model__C": [0.1, 1, 10, 100],"model__gamma": [0.001, 0.01, 0.1]}
svm_model = GridSearchCV(svm_pipe, svm_grid, scoring="roc_auc", cv=cv, n_jobs=1)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_prob = svm_model.predict_proba(X_test)[:, 1]
print("SVM Best Parameters:", svm_model.best_params_)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
log_pipe = Pipeline([("smote", SMOTE(random_state=123)), ("model", LogisticRegression(max_iter=5000))])
log_grid = {"model__C": [0.01, 0.1, 1, 10, 100], "model__penalty": ["l1", "l2"], "model__solver": ["liblinear"]}
log_model = GridSearchCV(log_pipe, log_grid, scoring="roc_auc", cv=cv, n_jobs=1)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_prob = log_model.predict_proba(X_test)[:, 1]

print("Logistic Best Parameters:", log_model.best_params_)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda_pipe = Pipeline([("smote", SMOTE(random_state=123)),("model", LinearDiscriminantAnalysis())])
lda_grid = {"model__solver": ["svd", "lsqr"]}
lda_model = GridSearchCV(lda_pipe, lda_grid,scoring="roc_auc",cv=cv, n_jobs=1)
lda_model.fit(X_train, y_train)
lda_pred = lda_model.predict(X_test)
lda_prob = lda_model.predict_proba(X_test)[:, 1]

print("LDA Best Parameters:", lda_model.best_params_)

#LightGBM
from lightgbm import LGBMClassifier

X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()

# Safe feature names
X_train_lgb.columns = [f"F{i}" for i in range(X_train_lgb.shape[1])]
X_test_lgb.columns = X_train_lgb.columns

# Pipeline
lgb_pipe = Pipeline([
    ("smote", SMOTE(random_state=123)),
    ("model", LGBMClassifier(random_state=123))
])

# Grid
lgb_grid = {
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__num_leaves": [15, 31, 63],
    "model__max_depth": [3, 5, 7]
}

# Grid Search
lgb_model = GridSearchCV(
    lgb_pipe,
    lgb_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1
)

# Train
lgb_model.fit(X_train_lgb, y_train)

# Prediction
lgb_pred = lgb_model.predict(X_test_lgb)
lgb_prob = lgb_model.predict_proba(X_test_lgb)[:, 1]

print("LGBM Best Parameters:", lgb_model.best_params_)

#Training predictions

rf_train_pred = rf_model.predict(X_train)
rf_train_prob = rf_model.predict_proba(X_train)[:, 1]
svm_train_pred = svm_model.predict(X_train)
svm_train_prob = svm_model.predict_proba(X_train)[:, 1]
log_train_pred = log_model.predict(X_train)
log_train_prob = log_model.predict_proba(X_train)[:, 1]
lda_train_pred = lda_model.predict(X_train)
lda_train_prob = lda_model.predict_proba(X_train)[:, 1]
lgb_train_pred = lgb_model.predict(X_train_lgb)
lgb_train_prob = lgb_model.predict_proba(X_train_lgb)[:, 1]

#Testing Predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:,1]
svm_pred = svm_model.predict(X_test)
svm_prob = svm_model.predict_proba(X_test)[:,1]
lda_pred = lda_model.predict(X_test)
lda_prob = lda_model.predict_proba(X_test)[:,1]
log_pred = log_model.predict(X_test)
log_prob = log_model.predict_proba(X_test)[:,1]
lgb_pred = lgb_model.predict(X_test_lgb)
lgb_prob = lgb_model.predict_proba(X_test_lgb)[:,1]
#Get metrics
import pandas as pd
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score)
def get_metrics(y_true, y_pred, y_prob, model_name, dataset):
    return pd.DataFrame({"Dataset":[dataset],"Model":[model_name],
        "Precision":[round(precision_score(y_true, y_pred, pos_label="T2D"), 2)],
        "Recall":[round(recall_score(y_true, y_pred, pos_label="T2D"), 2)],
        "F1":[round(f1_score(y_true, y_pred, pos_label="T2D"), 2)],
        "Accuracy":[round(accuracy_score(y_true, y_pred), 2)],
        "AUC":[round(roc_auc_score((y_true=="T2D").astype(int), y_prob), 2)]})
#Training Results
training_results = pd.concat([
    get_metrics(y_train, rf_train_pred, rf_train_prob, "RF", "Training"),
    get_metrics(y_train, svm_train_pred, svm_train_prob, "SVM", "Training"),
    get_metrics(y_train, lda_train_pred, lda_train_prob, "LDA", "Training"),
    get_metrics(y_train, log_train_pred, log_train_prob, "Logistic", "Training"),
    get_metrics(y_train, lgb_train_pred, lgb_train_prob, "LGBM", "Training")
], ignore_index=True)
#Testing Results
testing_results = pd.concat([
    get_metrics(y_test, rf_pred, rf_prob, "RF", "Testing"),
    get_metrics(y_test, svm_pred, svm_prob, "SVM", "Testing"),
    get_metrics(y_test, lda_pred, lda_prob, "LDA", "Testing"),
    get_metrics(y_test, log_pred, log_prob, "Logistic", "Testing"),
    get_metrics(y_test, lgb_pred, lgb_prob, "LGBM", "Testing")
], ignore_index=True)
#Final Result
final_results = pd.concat([training_results, testing_results],ignore_index=True)
print(final_results)
final_results.to_csv("SMOTE_Final_Model_Comparison.csv",index=False)


#95% CI

from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
def bootstrap_auc_ci(y_true, y_prob, n_bootstraps=2000, random_state=123):
    rng = np.random.RandomState(random_state)
    y_true = np.array((y_true == "T2D").astype(int))
    boot_scores = []
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        # Skip bootstrap samples containing only one class
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        boot_scores.append(score)
    lower = np.percentile(boot_scores, 2.5)
    upper = np.percentile(boot_scores, 97.5)
    auc = roc_auc_score(y_true, y_prob)
    return auc, lower, upper
auc_results = []
for name, prob in [
    ("RF", rf_prob),
    ("SVM", svm_prob),
    ("Logistic", log_prob),
    ("LDA", lda_prob),
    ("LGBM", lgb_prob)
]:
    auc, lower, upper = bootstrap_auc_ci(y_test, prob)
    auc_results.append({
        "Model": name,
        "AUC": round(auc, 3),
        "Lower95CI": round(lower, 3),
        "Upper95CI": round(upper, 3)})
auc_results = pd.DataFrame(auc_results)
auc_results.to_csv("Model_AUC_95CI.csv", index=False)
print(auc_results)

#ROC CURVE

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute ROC curves
fpr_rf, tpr_rf, _ = roc_curve((y_test=="T2D").astype(int), rf_prob)
fpr_svm, tpr_svm, _ = roc_curve((y_test=="T2D").astype(int), svm_prob)
fpr_log, tpr_log, _ = roc_curve((y_test=="T2D").astype(int), log_prob)
fpr_lda, tpr_lda, _ = roc_curve((y_test=="T2D").astype(int), lda_prob)
fpr_lgb, tpr_lgb, _ = roc_curve((y_test=="T2D").astype(int), lgb_prob)

# Calculate AUC
auc_rf = auc(fpr_rf, tpr_rf)
auc_svm = auc(fpr_svm, tpr_svm)
auc_log = auc(fpr_log, tpr_log)
auc_lda = auc(fpr_lda, tpr_lda)
auc_lgb = auc(fpr_lgb, tpr_lgb)

# Plot
plt.figure(figsize=(7,7))

plt.plot(fpr_rf, tpr_rf, lw=2, label=f'Random Forest (AUC = {auc_rf:.3f})')
plt.plot(fpr_svm, tpr_svm, lw=2, label=f'SVM (AUC = {auc_svm:.3f})')
plt.plot(fpr_log, tpr_log, lw=2, label=f'Logistic Regression (AUC = {auc_log:.3f})')
plt.plot(fpr_lda, tpr_lda, lw=2, label=f'LDA (AUC = {auc_lda:.3f})')
plt.plot(fpr_lgb, tpr_lgb, lw=2, label=f'LightGBM (AUC = {auc_lgb:.3f})')

plt.plot([0,1],[0,1],'k--',lw=1)

plt.xlabel("False Positive Rate", fontsize=13)
plt.ylabel("True Positive Rate", fontsize=13)
plt.title("ROC Curves", fontsize=15)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ROC_Curve.tiff", dpi=600)
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

# Precision-Recall
precision_rf, recall_rf, _ = precision_recall_curve((y_test=="T2D").astype(int), rf_prob)
precision_svm, recall_svm, _ = precision_recall_curve((y_test=="T2D").astype(int), svm_prob)
precision_log, recall_log, _ = precision_recall_curve((y_test=="T2D").astype(int), log_prob)
precision_lda, recall_lda, _ = precision_recall_curve((y_test=="T2D").astype(int), lda_prob)
precision_lgb, recall_lgb, _ = precision_recall_curve((y_test=="T2D").astype(int), lgb_prob)

ap_rf = average_precision_score((y_test=="T2D").astype(int), rf_prob)
ap_svm = average_precision_score((y_test=="T2D").astype(int), svm_prob)
ap_log = average_precision_score((y_test=="T2D").astype(int), log_prob)
ap_lda = average_precision_score((y_test=="T2D").astype(int), lda_prob)
ap_lgb = average_precision_score((y_test=="T2D").astype(int), lgb_prob)

plt.figure(figsize=(7,7))

plt.plot(recall_rf, precision_rf, lw=2, label=f'Random Forest (AP = {ap_rf:.3f})')
plt.plot(recall_svm, precision_svm, lw=2, label=f'SVM (AP = {ap_svm:.3f})')
plt.plot(recall_log, precision_log, lw=2, label=f'Logistic Regression (AP = {ap_log:.3f})')
plt.plot(recall_lda, precision_lda, lw=2, label=f'LDA (AP = {ap_lda:.3f})')
plt.plot(recall_lgb, precision_lgb, lw=2, label=f'LightGBM (AP = {ap_lgb:.3f})')
plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
plt.title("Precision–Recall Curves", fontsize=15)
plt.legend(loc="lower left", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Precision_Recall_Curve.tiff", dpi=600)
plt.show()
roc_data = pd.DataFrame({
    "Model": ["Random Forest", "SVM", "Logistic Regression", "LDA", "LightGBM"],
    "AUC": [auc_rf, auc_svm, auc_log, auc_lda, auc_lgb]})
roc_data.to_csv("ROC_AUC_Results.csv", index=False)
pr_data = pd.DataFrame({
    "Model": ["Random Forest", "SVM", "Logistic Regression", "LDA", "LightGBM"],
    "Average Precision": [ap_rf, ap_svm, ap_log, ap_lda, ap_lgb]})
pr_data.to_csv("PR_Average_Precision_Results.csv", index=False)

# ==========================================
# ROC AUC 95% CI (Training + Testing)
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Binary labels
y_train_bin = (y_train == "T2D").astype(int).values
y_test_bin = (y_test == "T2D").astype(int).values

# Bootstrap AUC 95% CI
def auc_ci_bootstrap(y_true, y_prob, n_bootstraps=2000, seed=123):

    rng = np.random.RandomState(seed)

    auc = roc_auc_score(y_true, y_prob)

    scores = []

    for i in range(n_bootstraps):

        idx = rng.randint(0, len(y_true), len(y_true))

        if len(np.unique(y_true[idx])) < 2:
            continue

        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return auc, lower, upper


# ==========================================
# TRAINING RESULTS
# ==========================================

train_models = {
    "Random Forest": rf_train_prob,
    "SVM": svm_train_prob,
    "Logistic Regression": log_train_prob,
    "LDA": lda_train_prob,
    "LightGBM": lgb_train_prob
}

train_results = []

plt.figure(figsize=(7,7))

for name, prob in train_models.items():

    fpr, tpr, _ = roc_curve(y_train_bin, prob)

    auc, lower, upper = auc_ci_bootstrap(y_train_bin, prob)

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{name} (AUC={auc:.3f},95%CI={lower:.3f}-{upper:.3f})"
    )

    train_results.append({
        "Model": name,
        "AUC": round(auc,3),
        "Lower95CI": round(lower,3),
        "Upper95CI": round(upper,3)
    })

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Training ROC Curves")
plt.legend(fontsize=8)
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("Training_ROC_with_95CI.tiff",dpi=600)
plt.show()

train_results = pd.DataFrame(train_results)
train_results.to_csv("Training_ROC_AUC_95CI.csv",index=False)

print(train_results)


# ==========================================
# TESTING RESULTS
# ==========================================

test_models = {
    "Random Forest": rf_prob,
    "SVM": svm_prob,
    "Logistic Regression": log_prob,
    "LDA": lda_prob,
    "LightGBM": lgb_prob
}

test_results = []

plt.figure(figsize=(7,7))

for name, prob in test_models.items():

    fpr, tpr, _ = roc_curve(y_test_bin, prob)

    auc, lower, upper = auc_ci_bootstrap(y_test_bin, prob)

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{name} (AUC={auc:.3f},95%CI={lower:.3f}-{upper:.3f})"
    )

    test_results.append({
        "Model": name,
        "AUC": round(auc,3),
        "Lower95CI": round(lower,3),
        "Upper95CI": round(upper,3)
    })

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Testing ROC Curves")
plt.legend(fontsize=8)
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("Testing_ROC_with_95CI.tiff",dpi=600)
plt.show()
plt.savefig("Training_ROC_with_95CI.png", dpi=600, bbox_inches="tight")
plt.savefig("Testing_ROC_with_95CI.png", dpi=600, bbox_inches="tight")
test_results = pd.DataFrame(test_results)
test_results.to_csv("Testing_ROC_AUC_95CI.csv",index=False)

print(test_results)
# ==========================================
# Precision-Recall Curve (Training + Testing)
# ==========================================

from sklearn.metrics import precision_recall_curve, average_precision_score

# Bootstrap AP 95% CI
def ap_ci_bootstrap(y_true, y_prob, n_bootstraps=2000, seed=123):

    rng = np.random.RandomState(seed)
    ap = average_precision_score(y_true, y_prob)
    scores = []

    for i in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(average_precision_score(y_true[idx], y_prob[idx]))

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return ap, lower, upper


# =========================
# TRAINING PR CURVE
# =========================

train_models = {
    "Random Forest": rf_train_prob,
    "SVM": svm_train_prob,
    "Logistic Regression": log_train_prob,
    "LDA": lda_train_prob,
    "LightGBM": lgb_train_prob
}

train_results = []

plt.figure(figsize=(7,7))

for name, prob in train_models.items():

    precision, recall, _ = precision_recall_curve(y_train_bin, prob)

    ap, lower, upper = ap_ci_bootstrap(y_train_bin, prob)

    plt.plot(
        recall,
        precision,
        lw=2,
        label=f"{name} (AP={ap:.3f},95%CI={lower:.3f}-{upper:.3f})"
    )

    train_results.append({
        "Model": name,
        "Average Precision": round(ap,3),
        "Lower95CI": round(lower,3),
        "Upper95CI": round(upper,3)
    })

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Training Precision-Recall Curves")
plt.legend(fontsize=8)
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("Training_PR_with_95CI.tiff", dpi=600)
plt.show()

pd.DataFrame(train_results).to_csv(
    "Training_PR_95CI.csv",
    index=False
)


# =========================
# TESTING PR CURVE
# =========================

test_models = {
    "Random Forest": rf_prob,
    "SVM": svm_prob,
    "Logistic Regression": log_prob,
    "LDA": lda_prob,
    "LightGBM": lgb_prob
}

test_results = []

plt.figure(figsize=(7,7))

for name, prob in test_models.items():

    precision, recall, _ = precision_recall_curve(y_test_bin, prob)

    ap, lower, upper = ap_ci_bootstrap(y_test_bin, prob)

    plt.plot(
        recall,
        precision,
        lw=2,
        label=f"{name} (AP={ap:.3f},95%CI={lower:.3f}-{upper:.3f})"
    )

    test_results.append({
        "Model": name,
        "Average Precision": round(ap,3),
        "Lower95CI": round(lower,3),
        "Upper95CI": round(upper,3)
    })

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Testing Precision-Recall Curves")
plt.legend(fontsize=8)
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("Testing_PR_with_95CI.tiff", dpi=600)
plt.show()
plt.savefig("Training_PR_with_95CI.png", dpi=600, bbox_inches="tight")
plt.savefig("Testing_PR_with_95CI.png", dpi=600, bbox_inches="tight")
pd.DataFrame(test_results).to_csv(
    "Testing_PR_95CI.csv",
    index=False
)

