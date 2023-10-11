# Dependent random variables
Previously we have look a probability distributions where things are independent, but real world data usually have some dependence that can be used for the compression.

## Dependence 
### Join entropy

$$H(X, Y) = \sum_{xy \in \mathcal{A}_x \mathcal{A}_y} = P(X, Y) log(\frac{1}{P(X, Y)}$$

Entropy is additive for independent variables i.e ($P(X,Y) = P(X)P(Y)$)
$$H(X,Y) = H(X) + H(Y)$$

### Conditional entropy $X$ given $y = b_k$
Cool, I didn't even know this was a [thing](https://en.wikipedia.org/wiki/Conditional_entropy), but it does make sense 

Some edge cases
| Equation | Meaning |
|--------|--------|
| $H(Y\|X) = 0$ |If and only $Y$ is completely determined by $X$|
| $H(Y\|X) = H(Y)$ |If and only $Y$ and $X$ is independent variables|

The definition itself is 
$$
H(X | y = b_k) \equiv  log (\frac{1}{P(X | y=b_k)})
$$

### Conditional entropy $X$ given $Y$
$$
H(X | Y) \equiv \sum_{y \in \mathcal{A}_Y} P(Y) \big[ \sum_{x \in \mathcal{A}_x} P(X|Y) log(\frac{1}{P(x | y)}) \big] = \sum_{xy \in \mathcal{A}_x \mathcal{A}_y}  P(X, Y) log(\frac{1}{P(X|Y)}
$$

This measures the remaining uncertainty of $X$ given $Y$

### Marginal entropy
The name of the entropy of $X$ ($HX(X)$) can also be called the marginal entropy to separate it from the conditional entropy.

### Chain rule for the information content
Product rule for probabilities
$$
log \frac{1}{P(X,Y)} = log\frac{1}{P(X)} + log\frac{1}{P(y | X)}
$$

So

$$
h(x,y) = h(x) + h(y)
$$

### Chain rule for entropy
$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

Which means that the uncertainty of $X$ and $Y$. 

### The mutual information
$$
I(X;Y) \equiv H(X) - H(X|Y)
$$

Which satisfies that $I(X;Y) = I(Y;X)$ and $I(X;Y) > 0$

How much did we learn about $X$ by knowing $Y$ ? 

### Mutual information between $X$ and $Y$ given $Z=c_k$
$$
I(X;Y | z = c_k) = H(X | z = c_k) - H(X | Y, z = c_k )
$$

**need to look more at this**

### The conditional mutual information between $X$ and $Y$ given $Z$
$$
I(X;Y | Z) = H(X | Z) - H(X | Y, Z )
$$

**need to look more at this**
´

----

There is a figure 8.1 in this chapter that is meant to show the relationship between these quantities, but would be nicer to write a script to see this. On [Wikipedia](https://en.wikipedia.org/wiki/Information_diagram) there is a relation diagram also.

![Diagram](https://i.stack.imgur.com/0m7XN.png)



