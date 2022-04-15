## Implementation of Open Entity Alignment (Open-EA)

Paper: Embedding-based Entity Alignment of Cross-lingual Temporal Knowledge Graphs(CTEA)



This repository contains the implementation of the CTEA architectures described in the paper.

## Installation
* Python 3.x (tested on Python 3.6)
* Tensorflow 1.x (tested on Tensorflow 1.8 and 1.12)
* Scipy
* Numpy
* Graph-tool or igraph or NetworkX
* Pandas
* Scikit-learn
* Matching==0.1.1
* Gensim

We recommend creating a new conda environment to install and run OpenEA. You should first install tensorflow-gpu (tested on 1.8 and 1.12), graph-tool (tested on 2.27 and 2.29,  the latest version would cause a bug), and python-igraph using conda:

```bash
conda create -n openea python=3.6
conda activate openea
conda install tensorflow-gpu==1.12
conda install -c conda-forge graph-tool==2.29
conda install -c conda-forge python-igraph
```

Then, OpenEA can be installed using pip with the following steps:

```bash
git clone https://github.com/nju-websoft/OpenEA.git OpenEA
cd OpenEA
pip install -e .
```
## Package Description
```
run/
├── openea/
│   ├── approaches/: package of the implementations for existing embedding-based entity alignment approaches
│   ├── models/: package of the implementations for unexplored relationship embedding models
│   ├── modules/: package of the implementations for the framework of embedding module, alignment module, and their interaction
│   ├── expriment/: package of the implementations for evalution methods
```

## Datasets
We choose three well-known KGs as our sources: DBpedia(English), DBpedia(French), and DBpedia(German). Besides,  the sparse relational dataset and the dense relational dataset are generated by the IDS algorithm. we generate two versions of datasets for each pair of KGs to be aligned. V1 is generated by directly using the IDS algorithm. For V2, we first randomly delete entities with low degrees (d <= 5) in the source KG to make the average degree doubled, and then execute IDS to fit the new KG. The statistics of the datasets are shown below.  


|---------Types---------|Languages|Dataset names|

|--------------------------|----------|----------------------| 

|15K sparse datasets |English&French| EN_ FR_ 15K_V1

|15K sparse datasets |English&German| EN_ DE_ 15K_V1

|15K dense datasets | English&French| EN_ FR_ 15K_V2

|15K dense datasets | English&German| EN_ DE_ 15K_V2

The  datasets can be downloaded from [Baidu Wangpan](https://pan.baidu.com/s/1mYec9tLp9tQpnqx0JsH7xw) (password: 85kq).

## Experiments

### Experiment Settings
The common hyper-parameters used for OpenEA are shown below.

<table style="text-align:center">
    <tr>
        <td style="text-align:center"></td>
        <th style="text-align:center">15K</th>
        
    </tr>
    <tr>
        <td style="text-align:center">Batch size for rel. triples</td>
        <td style="text-align:center">5,000</td>
        
    </tr>
    <tr>
        <td style="text-align:center">Termination condition</td>
        <td style="text-align:center" colspan="2">Early stop when the Hits@1 score begins to drop on <br>
            the validation sets, checked every 10 epochs.</td>
    </tr>
    <tr>
        <td style="text-align:center">Max epochs</td>
        <td style="text-align:center" colspan="2">2,000</td>
    </tr>
</table>

Besides, it is well-recognized to split a dataset into training(20%), validation(10%) and test(70%) sets. 
We use Hits@m (m = 1, 5, 10), mean rank (MR) and mean reciprocal rank (MRR) as the evaluation metrics.  Higher Hits@m and MRR scores as well as lower MR scores indicate better performance.

### Train and Test
To run the off-the-shelf approaches on our datasets and reproduce our experiments, change into the ./run/ directory and use the following script:


```bash
python main_from_args.py "predefined_arguments" "dataset_name" "split"
```

For example, if you want to run CTEA on EN_FR_15K_V1, please execute the following script:
    
```bash
python main_from_args.py ./args/ctea_args_15K.json EN_FR_15K_V1/721_5fold/1/
```

