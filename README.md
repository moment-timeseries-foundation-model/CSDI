# CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation

Official implementation of the paper "[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)" published in NeurIPS 2021.

CSDI is a novel diffusion model for probabilistic time series imputation that can:
- Handle irregular time intervals
- Process multivariate time series data
- Incorporate uncertainty estimates
- Condition on available observations

## Installation

### Requirements

Install the required packages with pip:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The repository supports experiments on two datasets:

### Healthcare Dataset (PhysioNet)

Download and preprocess the healthcare dataset with:

```bash
python download.py physio
```

### Air Quality Dataset (PM2.5)

Download and preprocess the air quality dataset with:

```bash
python download.py pm25
```

## Experiments

### Healthcare Dataset (PhysioNet)

#### Training and Imputation

To train a new model and run imputation on the healthcare dataset:

```bash
python exe_physio.py --testmissingratio [missing_ratio] --nsample [number_of_samples]
```

Parameters:
- `missing_ratio`: Ratio of missing values in test data (e.g., 0.1, 0.5)
- `number_of_samples`: Number of samples to generate for each missing value

### Air Quality Dataset (PM2.5)

To train and run imputation on the air quality dataset:

```bash
python exe_pm25.py --nsample [number_of_samples]
```

## Results Visualization

Use the Jupyter notebook `visualize_examples.ipynb` to visualize imputation results and uncertainty estimates.

## Acknowledgements

Parts of this implementation are based on:
- [BRITS](https://github.com/caow13/BRITS) (Bidirectional Recurrent Imputation for Time Series)
- [DiffWave](https://github.com/lmnt-com/diffwave) (Neural vocoder based on diffusion probabilistic models)

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## License

[MIT License](LICENSE)
