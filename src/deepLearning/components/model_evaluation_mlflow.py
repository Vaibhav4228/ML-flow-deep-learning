import tensorflow as tf
import mlflow
import tempfile
import shutil
from urllib.parse import urlparse
from pathlib import Path
from deepLearning.entity.config_entity import EvaluationConfig
from deepLearning.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        dg_kwargs = dict(rescale=1.0/255, validation_split=0.30)
        df_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        gen = tf.keras.preprocessing.image.ImageDataGenerator(**dg_kwargs)
        self.valid_generator = gen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **df_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(str(path), compile=False)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        scheme = urlparse(mlflow.get_tracking_uri()).scheme

        mlflow.log_params(self.config.all_params)
        mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

        
        tmpdir = tempfile.mkdtemp()
        export_path = Path(tmpdir) / "saved_model"
        self.model.export(str(export_path))

        
        run_kwargs = {"nested": True} if mlflow.active_run() else {}
        with mlflow.start_run(**run_kwargs) as run:
            mlflow.log_artifacts(str(export_path), artifact_path="model")
            if scheme != "file":
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri, "VGG16Model")

        shutil.rmtree(tmpdir)
