#!/home/pekandolf/.local/bin/pdm run
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from data import load_cats_vs_dogs
from myio import save_skops as save
import logging
from pathlib import Path

logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s',
                    level=logging.DEBUG)
X_train, y_train, X_test, y_test = load_cats_vs_dogs()

logging.debug("Create classifier")
voting_clf = make_pipeline(
    PCA(n_components=41),
    VotingClassifier(
        estimators=[
            ("lda", LinearDiscriminantAnalysis()),
            ("rf", RandomForestClassifier(
                n_estimators=500,
                max_leaf_nodes=2,
                random_state=6020)),
            ("svc", SVC(
                kernel="linear",
                probability=True,
                random_state=6020)),
        ],
        flatten_transform=False,
    )
)

logging.debug("Train classifier")
voting_clf.fit(X_train, y_train)

logging.debug("Score classifier")
score = voting_clf.score(X_test, y_test)

logging.info(f"We have a hard voting score of {score}")

dir = Path(".") / "models"
dir.mkdir(parents=True, exist_ok=True)

file = dir / "model"
save(voting_clf, file, X_train[:1])
