from PIL import Image


class Dataloader:
    def __init__(self) -> None:
        self.max = 3510
        self.speed = open("dataset/speed.txt", "r").read().split("\n")
        self.speed = list(map(float, self.speed))

    def load_image(self, idx):
        if (idx + 1) <= self.max:
            first_image = Image.open(f"dataset/frame{idx}.jpg")
            second_image = Image.open(f"dataset/frame{idx + 1}.jpg")
            miles_per_hour_to_meter_per_second = 2.237

            speed_1 = self.speed[idx] / miles_per_hour_to_meter_per_second
            speed_2 = self.speed[idx + 1] / miles_per_hour_to_meter_per_second

            # in meter ?
            distance = 1/2 * (speed_1 + speed_2) * (20/1000)

            return {
                "first_image": first_image,
                "second_image": second_image,
                "distance": distance
            }
        return None

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


if __name__ == "__main__":
    Dataloader().load_image(0)
