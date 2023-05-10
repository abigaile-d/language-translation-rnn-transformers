# Language Translation with RNN and Transformers

This project aims to develop a French-to-English translation program using two of the most common algorithms for sequence-to-sequence data: Recurrent Neural Networks (RNN) and Transformers. It also aims to compare the performance of these two architectures and identify which one works better for the translation task.


## Architecture

As previously mentioned, two separate translation models are trained using RNN and Transformers architectures. Both models use the sequence-to-sequence (Seq2Seq) architecture with an encoder-decoder approach. In this approach, the encoder processes the French sentence to produce hidden states, which are then used by the decoder to generate the English translation.

### RNNs

RNNs are a popular architecture for language translation tasks due to their ability to handle sequential data. The basic idea behind an RNN is to maintain a hidden state that represents the "memory" of the network. The hidden state is updated at each time step based on the current input and the previous hidden state.

__Pros__
- Can handle sequential data and capture long-term dependencies
- Can be trained using teacher forcing or sequence-to-sequence modeling
- Can be modified with different cell types, i.e. GRU for this implementation, to address the vanishing gradient problem

__Cons__
- Can be slow to train due to sequential nature and long-range dependencies
- Can suffer from the vanishing gradient problem, which can limit their ability to capture long-term dependencies
- Cannot handle input sequences of variable length

### Transformers

Transformers are a relatively new architecture for language translation tasks that have quickly become the state-of-the-art. Transformers are based on self-attention mechanisms that allow the network to attend to different parts of the input sequence at each time step.

__Pros__
- Can handle variable-length input sequences
- Can capture global dependencies between words and sentences
- Can be trained in parallel, which makes them faster to train than RNNs

__Cons__
- Can be more difficult to implement than RNNs
- May not perform as well on tasks that require modeling long-term dependencies


## Data

The dataset used in this project consists of over 8000 French-English sentence pairs for training and 2000 sentence pairs for testing purposes.

## Tech Stack
- Python
- Jupyter notebook
- PyTorch

## Results

The evaluation of the machine translation models is done using the BLEU (Bilingual Evaluation Understudy) score. This metric is commonly used to measure the quality of the translation output and is based on the comparison between the machine-generated translation and a set of reference translations. The BLEU score ranges from 0 to 1, where a score of 1 indicates a perfect match between the machine translation and the reference translations.

BLUE Scores:

| Model | Train BLEU Score | Test BLEU Score |
|-------|-----------------|-----------------|
| RNN   | 0.967 | 0.477 |
| Transformers | 0.939 | 0.588 |

The testing results based on BLEU score align with our expectations that the Transformer model would outperform the RNN model. The comparison of sample sentences translated by both models against the actual translation further confirms this observation. While there were some incorrect translations, many sentences were accurately translated, especially in the Transformers' results. Given the limited size of the dataset, the models' overall performance is commendable.


|    | Source                                      | Actual Translation                | RNN Translation                   | Transformer Translation           |
|---:|:--------------------------------------------|:----------------------------------|:----------------------------------|:----------------------------------|
|  0 | je suis en train de boire une biere .       | i m drinking a beer .             | i am drinking a letter .          | i am drinking a beer .            |
|  1 | elles cherchent un bouc emissaire .         | they re looking for a scapegoat . | they re looking for a scapegoat . | they re looking for a scapegoat . |
|  2 | ils ne constituent pas une menace .         | they re not a threat .            | they re not a bad good . .        | they re not watching .            |
|  3 | tu mens n est ce pas ?                      | you re lying aren t you ?         | you re staying aren t you ?       | you re lying aren t you ?         |
|  4 | je suis heureux de vous avoir invitee .     | i m glad i invited you .          | i m glad i invited you .          | i m glad i invited you .          |
|  5 | il connait le maire .                       | he is acquainted with the mayor . | he s open a chinese .             | he s stalling for tea .           |
|  6 | je suis interesse .                         | i m interested .                  | i m not .                         | i m interested .                  |
|  7 | nous sommes amoureux .                      | we re in love .                   | we re in .                        | we re biased .                    |
|  8 | elles sont chretiennes .                    | they are christians .             | they are christians .             | they are christians .             |
|  9 | je crains que tu m aies mal compris .       | i m afraid you misunderstood me . | i m afraid that will be happy .   | i m afraid you will get may .     |
| 10 | on a vraiment besoin d eau .                | we are badly in want of water .   | we re really proud of this .      | we re truly need .                |
| 11 | je suis toujours heureux .                  | i m always happy .                | i m always happy .                | i m always happy .                |
| 12 | tu es tout ce que j ai .                    | you re all i ve got .             | you re all i ve got .             | you re all i ve got .             |
| 13 | je mange un sandwich .                      | i m eating a sandwich .           | i am eating a sandwich .          | i m eating a sandwich .           |
| 14 | il est trop sensible .                      | he is too sensitive .             | he s too drunk .                  | he s too sensitive .              |
| 15 | vous etes prevenant .                       | you re considerate .              | you re considerate .              | you re considerate .              |
| 16 | j y vais .                                  | i m going .                       | i m going going .                 | i m going there .                 |
| 17 | on n est jamais trop vieux pour apprendre . | you re never too old to learn .   | he s too old to learn too old .   | you are too old to learn .        |
| 18 | elle prepare le dejeuner .                  | she is making dinner .            | she s missed to the . .           | she is making dinner .            |
| 19 | nous sommes tous en train de diner .        | we re all having lunch .          | we re all in . .                  | we re all growing .               |