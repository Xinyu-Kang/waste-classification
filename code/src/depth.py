class DepthModel():
    def __init__(self, config) -> None:
        self.config = config
        self.model = None
        self.load()

    def load(self):
        pass

    def predict(self, image, segmentation_results):
        pass