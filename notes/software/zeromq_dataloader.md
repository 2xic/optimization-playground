# [ZeroMq ](https://github.com/commaai/research/blob/1a437a954d0203da67dab4413ded3beb05bc6066/server.py#L101)

Comma uses(or used?) a ZeroMq message server to get a dataloader to load dataset from a master server to the research computers. Which is a great way of solving this issue instead of using more complicated [1][2] solutions. The ZMQ implementation posted above is less than 200 lines with both the server and client data.

The implementation hey used are based on [this](https://github.com/mila-iqia/fuel)


[1] https://github.com/spotify/luigi 
[2] https://docs.celeryq.dev/en/stable/
