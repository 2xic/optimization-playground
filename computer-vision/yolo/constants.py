
class Constants:
    def __init__(self,
                 BOUNDING_BOX_COUNT=2,
                 CLASSES=20
                 ) -> None:
        self.GRID_SIZE = 7
        self.BOUNDING_BOX_COUNT = BOUNDING_BOX_COUNT
        self.CLASSES = CLASSES  # 20

        self.tensor_grid_size = self.GRID_SIZE * self.GRID_SIZE * \
            (self.BOUNDING_BOX_COUNT * 5 + self.CLASSES)

        self.image_width = 500
        self.image_height = 500
