# Versatile User Identification in Extended Reality Using Pretrained Similarity-Learning

This repository contains the code for our paper ["Versatile User Identification in Extended Reality Using Pretrained Similarity-Learning"](https://arxiv.org/abs/2302.07517).

## About

This work combines distance-based and classification-based approaches to identify VR users from their movements using deep metric learning. The models are trained on data from players of "Half-Life: Alyx" and demonstrate:

- Ability to identify new users from non-specific movements with minimal enrollment data
- Fast new user enrollment (seconds vs days for retraining traditional classifiers) 
- More reliable performance with limited enrollment data
- Cross-dataset generalization to different VR devices

## Repository Structure

The codebase is organized into `data_preparation` and `machine_learning`. You find in each folder the corresponding Readmes.

## Citation

If you use this code in your research, please cite:

```bibtex
@online{RackVersatileUserIdentification2023,
  title = {Versatile {{User Identification}} in {{Extended Reality}} Using {{Pretrained Similarity-Learning}}},
  author = {Rack, Christian and Kobs, Konstantin and Fernando, Tamara and Hotho, Andreas and Latoschik, Marc Erich},
  date = {2023-07-03},
  eprint = {2302.07517},
  eprinttype = {arXiv},
  doi = {10.48550/arXiv.2302.07517}
}
```

## License

This work by Christian Rack, Konstantin Kob, Tamara Fernando, Andreas Hotho and Marc E. Latoschik is licensed under CC BY-NC-SA 4.0.