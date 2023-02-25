import abc

class GanLoss(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def generator(self,  discriminator, real, y, generator, noise):
        pass

    @abc.abstractclassmethod
    def discriminator(self, discriminator, real, y, generator, noise):
        pass
