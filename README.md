# VGL-GAN
## Video Game Level Generation using Deep Convolutional Generative Adversarial Network

## Proposed Model Architecture
![Model-Design](https://github.com/myaseenml/VGL-GAN/blob/main/Docs/Model.png)

## Generated Levels
![Model-Design](https://github.com/myaseenml/VGL-GAN/blob/main/Docs/Levels.png)

-----

## Getting Started

Download or clone this repository on your system.

### Prerequisites
```
- PYTHON 3.8.1
- JAVA 1.8.0
```

### Training the GAN
The GAN that generates Mario levels can be run by the following command in the GANTrain folder:

```
python3 GANTraining.py --cuda
```

### Running LSI Experiments
Experiments can be run with the command:
```
python3 search/run_search.py -w 1 -c search/config/experiment/experiment.tml
```

The w parameter specifies a worker id which specifies which trial to run from a given experiment file. This allows for parallel execution of all trials on a high-performance cluster.

## Future Work
> - Extend this tool for other video games like DOOM and Lode Runner

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
