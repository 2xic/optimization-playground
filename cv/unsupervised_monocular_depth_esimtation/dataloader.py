

class Dataloader:
    def __init__(self) -> None:
        pass

    def get_pair(self, x, y):
        """
        We are a single camera.

        x -> y -> z
             y  <- < 

        Wait, in the paper only left is used
            -> Left is applied to model
            -> Model knows the length between
                -> This will be dynamic in our case because of speed...
            -> So we have the formula
                d = b * f / d_

                b = baseline distance between cameras
                    -> We can calculate this based on the speed of car
                f = camera focal length 
                    -> Not sure exactly
                    -> I think it's a eon (oneplus 3)
                    -> https://www.devicespecifications.com/en/model/437a3c9b
                d_ = disparity
                    https://stackoverflow.com/a/17620159          
        """
        pass


