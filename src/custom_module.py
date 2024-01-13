import mlflow.pyfunc
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    # def load_context(self, context):
    #     # Load any necessary artifacts or dependencies
    #     self.model = context.artifacts["model"]

    def predict(self, context, model_input):
      return self.model.predict(model_input)