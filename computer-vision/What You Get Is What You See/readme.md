Basic implementation (proof of concept) of the idea laid out in the paper.

Various things could be done to fix various problems with this implementation, for isntance the tokenizer is no good, it's one char based instead of symbol.


## Output 
After a few epcohs the model starts to give resaonble outputs :')

```
Output: $(r_ir_j)^\infty$
Expected: $(r_ir_j)^\infty$
Loss: 150.51931953430176

Output: $au+bv=1$
Expected: $au+bv=1$
Loss: 289.29118824005127

Output: $fhfrac{h_3^2 + h_2^2}{h_3} = \frac{h_3}{h_3 h_0 - h_2 h_1}$
Expected: $-\frac{h_3^2 + h_2^2}{h_3} = \frac{h_3}{h_3 h_0 - h_2 h_1}$
Loss: 150.08274269104004

Output: $1  \ph  27 + 178) \div (122 = 161 - 169 $
Expected: $76 \pm (27 - 168) \div (192 + 138 - 159)$
Loss: 201.56682014465332
```
