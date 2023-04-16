### [What You Get Is What You See: A Visual Markup Decompiler](https://arxiv.org/pdf/1609.04938v1.pdf)

Model that achives great results on going from Latex OCR to latex, and is trained end to end with convs and attention.
The input is a image, and the output is a sequence of text tokens (the markup language). 
Each row in a image will be encoded with an RNN that recieves a feature grid from the CONV, and then later decoded with another RNN with an attention mechanism.
