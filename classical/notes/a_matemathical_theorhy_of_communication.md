# [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
Communication system
1. Information sources goes to a transmitter that then becomes a signal
2. The transmitted signal cna have noise applied to it (bit flip over the wire etc)
3. The received signal is then decoded by the deceiver


# Khan academy has some great videos on this
[ A mathematical theory of communication | Computer Science | Khan Academy ](https://www.youtube.com/watch?v=WyAtOqfCiBw)

[ Information entropy | Journey into information theory | Computer Science | Khan Academy ](https://www.youtube.com/watch?v=2s3aJfRr9gE)
- Entropy -> To predict the naext symbol from a system, what is the minimal yes / no question you would expect to ask ? (paraphrase)
- ^ measure of average uncertainty is the definition Shanon used.  
- Entropy is max when all outcomes are equal (no information is revealed)
- Entropy goes down when predicable is inserted.
- Bit = measure of surprise / entropy 

# other videos
[ What is the Shannon capacity theorem? ](https://www.youtube.com/watch?v=ancDN11C2vg)

$$C= B \cdot log_2 (1 + \frac{s}{n})$$
Where $B$ is bandwidth, $s$ is signal and $n$ is noise. If `N` goes to infinity then $log_2$ will results in a channel capacity of "0". More bandwidth can help recover from noise.


