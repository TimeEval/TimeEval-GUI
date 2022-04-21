<div align="center">
<img width="100px" src="timeeval-icon.png" alt="TimeEval logo"/>
<h1 align="center">TimeEval GUI / Toolkit</h1>
<p>
A Benchmarking Toolkit for Time Series Anomaly Detection Algorithms
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![python version 3.7|3.8|3.9](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)

</div>

> If you use our artifacts, please consider [citing our papers](#Citation).

This repository hosts an extensible, scalable and automatic benchmarking toolkit for time series anomaly detection algorithms.
TimeEval includes an extensive data generator and supports both interactive and batch evaluation scenarios.
With our novel toolkit, we aim to ease the evaluation effort and help the community to provide more meaningful evaluations.

The following picture shows the architecture of the TimeEval Toolkit:

<div align="center">

![TimeEval architecture](./doc/figures/timeeval-architecture.png)

</div>

It consists of four main components: a visual frontend for interactive experiments, the Python API to programmatically configure systematic batch experiments, the dataset generator GutenTAG, and the core evaluation engine (Time)Eval.
While the frontend is hosted in this repository, GutenTAG and Eval are hosted in separate repositories.
Those repositories also include their respective Python APIs:

[![GutenTAG Badge](https://img.shields.io/badge/Repository-GutenTAG-blue?style=for-the-badge)](https://github.com/HPI-Information-Systems/gutentag)
[![Eval Badge](https://img.shields.io/badge/Repository-Eval-blue?style=for-the-badge)](https://github.com/HPI-Information-Systems/timeeval)

As initial resources for evaluations, we provide over 1,000 benchmark datasets and an increasing number of time series anomaly detection algorithms (over 70): 

[![Datasets Badge](https://img.shields.io/badge/Repository-Datasets-3a4750?style=for-the-badge)](https://hpi-information-systems.github.io/timeeval-evaluation-paper/notebooks/Datasets.html)
[![Algorithms Badge](https://img.shields.io/badge/Repository-Algorithms-3a4750?style=for-the-badge)](https://github.com/HPI-Information-Systems/TimeEval-algorithms)

## Installation and Usage (tl;dr)

### Web frontend

```shell
# install all dependencies
make install

# execute streamlit and display frontend in default browser
make run
```

Screenshots of web frontend:

![GutenTAG page](./doc/figures/gutentag.png)
![Eval page](./doc/figures/eval.png)
![Results page](./doc/figures/results.png)

### Python APIs

Install the required components using pip:

```bash
# eval component:
pip install timeeval

# dataset generator component:
pip install timeeval-gutentag
```

For usage instructions of the respective Python APIs, please consider the project's documentation:

[![GutenTAG Badge](https://img.shields.io/badge/Repository-GutenTAG-blue?style=for-the-badge)](https://github.com/HPI-Information-Systems/gutentag)
[![Eval Badge](https://img.shields.io/badge/Repository-Eval-blue?style=for-the-badge)](https://github.com/HPI-Information-Systems/timeeval)

## Citation

If you use the TimeEval toolkit or any of its components in your project or research, please cite our demonstration paper:

> tbd

If you use our evaluation results or our benchmark datasets and algorithms, please cite our evaluation paper:

> tbd
