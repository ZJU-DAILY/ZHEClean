# ZHEClean: Cleaning Dirty Knowledge Graphs using Zero Human-labeled Examples

ZHEClean, a novel cleaning framework powered by knowledge graph embedding, to not only detect but also repair dirty triples in knowledge graphs.
First, ZHEClean presents TransVAE, which is designed to collaboratively learn a trust-aware knowledge representation (TKR) model and a self-supervised triple classification (STC) model directly from a real-world knowledge graph that might contain errors. TransVAE does not require any labor-intensive label annotations. In addition, it guarantees the excellent performance of the knowledge representation model as well as that of the triple classification model via jointly training the two models in an iterative manner. 
Next, ZHEClean utilizes a propagation power-based PROR strategy to repair errors. 

For more technical details, see the ZHEClean: Cleaning Dirty Knowledge Graphs using Zero Human-labeled Examples paper.

![framework](framework.jpg)

## Requirements

* Python 3.7
* PyTorch 1.7.1
* CUDA 11.0

## Datasets

We conduct experiments on four representative and widely-used KG benchmarks:

- FB15K-237: a new version of FB15K, a subset of Freebase. 
- WN18RR: a knowledge graph extracted from the English lexical database WordNet.
- YAGO3-10-DR: a subset of a multilingual knowledge graph YAGO3 deriving from Wikipedia and WordNet.
- NELL27K: a real-world KG extracted from NELL.

The dataset configurations can be found in ``configs.json``. 

**Data format:**

- *entities.dict*: a dictionary map entities to unique ids.
- *relations.dict*: a dictionary map relations to unique ids.
- *train.txt*: the model is trained to fit this data set.
- *noise_xx.txt*: noise triples in the train set with a rate of xx.
- *valid.txt*: a part of the ground truth.
- *test.txt*: the model is evaluated on this data set.
- *test_negative_xx.txt*: noise triples in the test set with a rate of xx.

## KGClean

To train the knowledge graph embedding and error detection with TransVAE:

**TransVAE-E**

```
python run.py --noise_rate 20 --mode soft --data_name FB15K-237 --model TransE
```

**TransVAE-R**

```
python run.py --noise_rate 20 --mode soft --data_name FB15K-237 --model RotatE
```

To run the error repairing with KGClean:

**ZHEClean-E**

```
python erpair_error.py --data_name FB15K-237 --model TransE
```

**ZHEClean-R**

```
python erpair_error.py --data_name FB15K-237 --model RotatE
```

## Acknowledgement

We use the code of [OTE](https://github.com/JD-AI-Research-Silicon-Valley/KGEmbedding-OTE) and [semi-supervised-pytorch](https://github.com/wohlert/semi-supervised-pytorch).
