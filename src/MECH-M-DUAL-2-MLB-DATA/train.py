#!/home/pekandolf/.local/bin/pdm run
from model import get_model
from data import load_cats_vs_dogs
from myio import save_skops as save
import logging
from pathlib import Path
from omegaconf import OmegaConf

logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s',
                    level=logging.DEBUG)
X_train, y_train, X_test, y_test = load_cats_vs_dogs()

logging.debug("Load config")
params_path = Path(".") / "params.yaml"
params = OmegaConf.load(params_path)

components = [params["PCA"], params["VotingClassifier"]]
estimators = [params[i] for i in components[1].estimators]
voting_clf = get_model(components, estimators)

logging.debug("Train classifier")
voting_clf.fit(X_train, y_train)

logging.debug("Score classifier")
score = voting_clf.score(X_test, y_test)

logging.info(f"We have a hard voting score of {score}")

dir = Path(".") / "models"
dir.mkdir(parents=True, exist_ok=True)

file = dir / "model"
save(voting_clf, file, X_train[:1])
