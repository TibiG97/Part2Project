from abc import abstractmethod


class Classifier(object):

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def train(self,
              training_set,
              labels):
        pass

    @abstractmethod
    def predict_class(self,
                      test_set):
        pass

    @abstractmethod
    def predict_probs(self,
                      test_set):
        pass

    @abstractmethod
    def save_model(self,
                   model_path: str):
        pass

    @abstractmethod
    def load_model(self,
                   model_path: str):
        pass
