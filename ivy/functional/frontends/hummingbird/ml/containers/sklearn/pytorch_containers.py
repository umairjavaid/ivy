

from .._sklearn_api_containers import (
    SklearnContainer,
    SklearnContainerRegression,
    SklearnContainerClassification
)


# PyTorch containers.
class PyTorchSklearnContainer(SklearnContainer):
    """
    Base container for PyTorch models.
    """

    def save(self, location):
        raise NotImplemented

    @staticmethod
    def load(location, do_unzip_and_model_type_check=True, delete_unzip_location_folder: bool = True, digest=None):
        raise NotImplemented

class PyTorchSklearnContainerRegression(SklearnContainerRegression, PyTorchSklearnContainer):
    """
    Container for PyTorch models mirroring Sklearn regressor API.
    """

    def _predict(self, *inputs):
        if self._is_regression:
            output = self.model.forward(*inputs).cpu().numpy()
            if len(output.shape) == 2 and output.shape[1] > 1:
                # Multioutput regression
                return output
            else:
                return output.ravel()
        elif self._is_anomaly_detection:
            return self.model.forward(*inputs)[0].cpu().numpy().ravel()
        else:
            #print(self.model._forward(*inputs))
            return self.model._forward(*inputs)[0][0].to("cpu").numpy().ravel()
        
class PyTorchSklearnContainerClassification(SklearnContainerClassification, PyTorchSklearnContainerRegression):
    """
    Container for PyTorch models mirroring Sklearn classifiers API.
    """

    def _predict_proba(self, *input):
        return self.model.forward(*input)[1].cpu().numpy()