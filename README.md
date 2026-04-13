# Static and Dynamic Graph Alignment Network for Temporal Video Grounding

<p align="center">
  <a href="./README.md"><img alt="README in English" src="https://img.shields.io/badge/English-DBEDFA"></a>
  <a href="./README_zh.md"><img alt="简体中文版自述文件" src="https://img.shields.io/badge/简体中文-DFE0E5"></a>
</p>


<summary><b>📕 Table of Contents</b></summary>

- [Task Definition](#task-definition)
- [Innovation](#innovation)
- [Data Preparation](./data_preparation/data_preparation_README.md)
- [Environment Setting](./environment/env_README.md)
- [Training](#training)
- [Inference](#inference)
- [Main Results](#main-results)
- [More Information](#more-information)


## Task Definition

Temporal Video Grounding (TVG) also known as: Temporal sentence grounding in videos (TSGV), natural language video localization (NLVL), video moment retrieval (VMR).

The objective of **Temporal Video Grounding (TVG)** is to
localize the precise moment in an untrimmed video that
corresponds to given language queries.

![](./images/TVG_definition.png)

## Innovation

![the proposed SDGAN](./images/1b_0313.png)

The figure above provides a simple illustration of the proposed **Static and Dynamic Graph Alignment Network (SDGAN)**. The model introduces two key innovations:

- **Effective exploitation of both static and dynamic visual features**:  
  Unlike existing methods that typically rely on either static or dynamic visual features alone, SDGAN successfully integrates both. It employs several techniques — including **Position-wise Nodes Alignment** (as shown in the figure) — to effectively exploit and mitigate the semantic discrepancy between static and dynamic features. 

- **Query-aware video feature construction**:  
  GCN-based approaches construct temporal graphs in a query-agnostic manner, where video features interact according to predefined rules without guidance from the video content or the specific query.  
  To address this limitation, SDGAN introduces **Query–Clip Contrastive Learning** and **Adaptive Graph Modeling** (illustrated in the figure). This query-aware approach produces more discriminative node representations, leading to stronger temporal graph modeling.


The detailed mechanisms are presented in the paper.
## Training

## Inference

## Main Results

## More Information

### Acknowledgements
This code is inspired by [UniSDNet](https://github.com/xian-sh/UniSDNet), [ReLoCLNet](https://github.com/26hzhang/ReLoCLNet) [OSGNet](https://github.com/Yisen-Feng/OSGNet). We thank the authors for their awesome open-source contributions.

### Contact
If there are any questions, feel free to open an issue or contact the author: Zhanjie Hu (ZhanJieHu@163.com)