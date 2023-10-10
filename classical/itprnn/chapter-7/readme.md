# Code for integers


## Unary codes
- Send n zeros then a one to mark the end
- The length of the unary code should be l_u(n) = n

## Self-delimiting codes
- First sending the encoded version of the length and then sending the value
- That is first the length of the numbers with zeros and then the classical binary representation

## Code with end-of-file-symbol-
- We can send a byte-based representation
- Using 4 bits and base16 you can have `1111` as the end-of-file symbol


## Resources
- [Elias gamma coding](https://en.wikipedia.org/wiki/Elias_gamma_coding)
- [Integer encoding](http://didawiki.cli.di.unipi.it/lib/exe/fetch.php/magistraleinformaticanetworking/ae/ae2012/chap9.pdf)
