#!/home/pekandolf/.local/bin/pdm run
from data import load_cats_vs_dogs
from myio import save_skops as save
import model
import logging
from pathlib import Path
from omegaconf import OmegaConf
from dvclive import Live


def param_from_yaml(live, component):
    prefix = component.type.split(".")[-1]
    for name, value in component.init_args.items():
        live.log_param(prefix + "/" + name, value)


logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s',
                    level=logging.INFO)
X_train, y_train, X_test, y_test = load_cats_vs_dogs()

logging.debug("Load config")
config_path = Path(".") / "config.yaml"
config = OmegaConf.load(config_path)
components = config.model.components
estimators = components[1].init_args.estimators

logging.debug("Create classifier")
voting_clf = model.create(components, estimators)

with Live() as live:

    param_from_yaml(live, components[0])
    for est in estimators:
        param_from_yaml(live, est)

    logging.debug("Train classifier")
    voting_clf.fit(X_train, y_train)

    metrics = model.evaluate(voting_clf, [[X_train, y_train, "train"],
                                          [X_test, y_test, "test"]])
    for metric_name, value in metrics.items():
        live.log_metric(metric_name, value)

    y_pred = voting_clf.predict(X_test)
    live.log_sklearn_plot("confusion_matrix", y_test, y_pred)

    logging.info(f"We have a hard voting score of {metrics["test_score"]}")

    dir = Path(live.dir) / "artifacts"
    dir.mkdir(parents=True, exist_ok=True)

    file = dir / "model.skops"
    save(voting_clf, file, X_train[:1])
    live.log_artifact(file.relative_to(Path(".")), type="model")
