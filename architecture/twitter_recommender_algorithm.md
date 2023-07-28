
## [Blog post](https://blog.twitter.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm)
Three main stages in pipeline

1. Fetch tweets
2. Rank tweets
3. Filter out inappropriate tweets

Looking at the image that goes in a bit more details
1. Get the data
    - Follow graph
    - Tweet engagements
    - User data
2. Get features
    - SimCluster
    - TwHin
    - RealGraph
    - Trust and Safety
3. Candidate sources
    - In network (people you follow)
        - Most important component is [Real graph](https://www.ueo-workshop.com/wp-content/uploads/2014/04/sig-alternate.pdf)
    - Out network (people in your social graph)
        - Based on tweets that the people you follow have recently interacted with
        - In addition to what tweets similar users to yourself have been interacting with
        - [GraphJet](https://www.vldb.org/pvldb/vol9/p1281-sharma.pdf) is an important component here
        - [SimCluster](https://dl.acm.org/doi/pdf/10.1145/3394486.3403370)
4. Ranking
    - Done with a NN 
    - Online trained
    - 
5. Heuristic filtering
    - Social proof
        -> Make sure some in your social networks approve the tweets (in other words, have engaged with them)
    - Visibility (T&S) 
        -> Authors you have blocked for instance
    - Author diversity
        -> Don't have to many tweets form the same author
    - Content balance
        -> Balance in network and out network tweets
    - Feedback fatigue
        -> Down rank tweets with bad feedback
6. This is then fed into a mixer with
    - Rank from previous steps
    - Ads
    - Who to follow
7. Timeline is created :)


## [Source code](https://github.com/twitter/the-algorithm)
Look at the various parts of the stack, looks like good resource. 

### General
- Interesting that most of the code is in scala and Java
  - Because this fetches the features and data pipeline
  - https://github.com/twitter/the-algorithm/blob/main/tweetypie/server/README.md
  - 

### Models
Looking at some of the models
  - https://github.com/twitter/the-algorithm/tree/main/trust_and_safety_models
    - Most of the training scripts are "single purpose" (not modular at all)
  - [SimCluster](https://www.kdd.org/kdd2020/accepted-papers/view/simclusters-community-based-representations-for-heterogeneous-recommendatio)
    - Published and has the job of finding similar communities based on embeddings
  - [Tweepcred](https://github.com/twitter/the-algorithm/blob/main/src/scala/com/twitter/graph/batch/job/tweepcred/README)
    - Modified pagerank for Twitter reputation

