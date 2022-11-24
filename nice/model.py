from layer import AddictiveCouplingLayer

class Model:
    def __init__(self) -> None:
        self.layer_1 = AddictiveCouplingLayer()
        self.layer_2 = AddictiveCouplingLayer()
        self.layer_3 = AddictiveCouplingLayer()
        self.layer_4 = AddictiveCouplingLayer()
    
    def forward(self, x_1, x_2):
        (h_1_1, h_1_2) = self.layer_1.forward(x_1, x_2)
        (h_2_1, h_2_2) = self.layer_1.forward(
            h_1_1, 
            x_2,
            prev=h_1_2
        )
        (h_3_1, h_3_2) = self.layer_1.forward(
            h_2_1, 
            x_1,
            prev=h_2_2
        )
        (h_4_1, h_4_2) = self.layer_1.forward(
            h_3_1, 
            x_2,
            prev=h_3_2
        )
        return (
            # exp(s) + h
        )
