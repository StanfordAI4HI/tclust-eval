# Evaluation Criteria for Temporal Clustering Evaluation

This repository provides an implementation of new evaluation criteria introduced in,
 
**Learning Procedural Abstractions and Evaluating Discrete Latent Temporal Structure**  
Karan Goel and Emma Brunskill  
_ICLR 2019_
 

#### What's included
- New evaluation criteria (Repeated Structure Score, Label Agnostic Segmentation Score, Segment Structure Score, 
Temporal Structure Score)
- Code for visualizing time-series segmentations.
 

#### Who should use this
The evaluation criteria are meant for research into unsupervised learning 
methods that segment time-series or do changepoint detection.

If you're doing unsupervised segmentation of time-series and want to compare to 
ground-truth segmentations, then this is a substitute for using common criteria such as the normalized mutual information or 
Munkres (Hungarian method) score. 

It is also possible to use the label agnostic scores in supervised learning applications to understand how accurately
your method is finding changepoints (e.g. switching to a different activity in vision applications).

#### Setup and Usage

Install the requirements into your virtual environment using ``pip install -r requirements.txt``. For example usage,
 see the Jupyter notebook ``Tutorial.ipynb``.