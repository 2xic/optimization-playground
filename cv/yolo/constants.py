
class Constants:
    def __init__(self) -> None:
        self.GRID_SIZE = 7 
        self.BOUNDING_BOX_COUNT = 2
        self.CLASSES = 20

        self.grid_size = self.GRID_SIZE * self.GRID_SIZE + \
            (self.BOUNDING_BOX_COUNT * 5 + self.CLASSES)

        """
            We know the output format
        """