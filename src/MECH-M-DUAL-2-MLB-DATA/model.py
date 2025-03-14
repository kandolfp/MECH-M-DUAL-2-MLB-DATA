import logging
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


def evaluate(clf, data, cv=5) -> dict:
    metrics = {}
    for X, y, prefix in data:
        logging.debug(f"Score classifier for {prefix}")
        score = clf.score(X, y)
        metrics[f"{prefix}_score"] = score
    return metrics


def create(components, estimators):
    voting_clf = make_pipeline(
        PCA(**components[0].init_args),
        VotingClassifier(
            estimators=[
                ("lda", LinearDiscriminantAnalysis(
                    **estimators[0].init_args
                )),
                ("rf", RandomForestClassifier(
                    **estimators[1].init_args)),
                ("svc", SVC(
                    **estimators[2].init_args)),
            ],
            flatten_transform=components[1].init_args.flatten_transform,
        )
        )
    return voting_clf
