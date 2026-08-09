# Contributing to Dynamic Graph Neural Network-Based Intrusion Detection for IoT Networks

Thank you for your interest in contributing to this project! We welcome contributions that improve the codebase, documentation, testing, performance, and overall reliability of the intrusion detection pipeline.

This guide explains how to set up the development environment, run tests, follow the project’s coding standards, and submit changes.

---

## Getting Started

### 1. Fork and Clone the Repository

Fork this repository to your GitHub account and clone your fork locally:

```bash
git clone https://github.com/<your-username>/Dynamic-Graph-Neural-Network-Based-Intrusion-Detection-for-IoT-Networks.git
cd Dynamic-Graph-Neural-Network-Based-Intrusion-Detection-for-IoT-Networks
```

### 2. Create a Virtual Environment

Create and activate a Python virtual environment:

**Linux/macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

> **GPU/CUDA users:** Please review the note at the top of `requirements.txt`. PyTorch and PyTorch Geometric may need to be installed separately using the appropriate versions for your CUDA environment before installing the remaining dependencies.

### 3. Download the Dataset

This project uses the **CICIoT2023** dataset for IoT network intrusion detection.

Download the dataset from the official source:

* [CICIoT2023 Dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

Place the extracted dataset under:

```text
data/raw/CICIoT2023/
```

Follow the dataset organization and folder structure described in the project's `README.md`.

> **Note:** The CICIoT2023 dataset is not included in this repository because of its large size and is excluded through `.gitignore`.

### 4. Run the Smoke Test

Before making changes, verify that your development environment is configured correctly by running a small pipeline test:

```bash
python main.py --stage all --nrows 2000
```

Using `--nrows 2000` limits the number of rows processed and allows you to quickly verify the complete pipeline without processing the full dataset.

---

## Running Tests

Run the complete test suite with:

```bash
pytest tests/ -v
```

The test suite currently contains **26 tests** and uses small synthetic datasets. Therefore, the tests can be executed without downloading the full CICIoT2023 dataset.

Before submitting a contribution, please ensure that:

* All existing tests pass.
* New functionality includes appropriate tests.
* Existing functionality is not unintentionally broken.
* The test suite completes without errors or unexpected warnings.

---

## Project Structure

The project is organized into modular components. A high-level overview is provided below:

```text
src/
├── data/          # Data loading, cleaning, and feature selection
├── graph/         # k-NN similarity graph construction
├── models/        # GCN, GraphSAGE, GAT, and classical ML models
├── training/      # Model training and evaluation
└── utils/         # Metrics, visualization, and reusable utilities
```

For the complete directory structure and pipeline description, please refer to the project's `README.md`.

### Module Responsibilities

| Module          | Responsibility                                                  |
| --------------- | --------------------------------------------------------------- |
| `src/data/`     | Dataset loading, preprocessing, cleaning, and feature selection |
| `src/graph/`    | Similarity graph and k-NN graph construction                    |
| `src/models/`   | Graph neural networks and classical ML baselines                |
| `src/training/` | Model training, validation, and evaluation                      |
| `src/utils/`    | Metrics, plotting, and reusable helper functions                |

Please maintain these module boundaries when adding new functionality.

---

## Code Style and Quality

To keep the project maintainable and consistent, contributors should follow these guidelines:

* Follow the existing coding style and project conventions.
* Use **type hints** for function signatures.
* Add **docstrings** to every new public function, class, and module where appropriate.
* Use descriptive and meaningful variable and function names.
* Keep functions focused on a single responsibility.
* Avoid unnecessary duplication of code.
* Keep data loading, preprocessing, graph construction, modeling, and evaluation logic separated.
* Place functionality in the appropriate module rather than introducing unrelated responsibilities into existing files.
* Add comments only where they provide useful context that is not obvious from the code itself.

For example, scaling and preprocessing logic should remain in `preprocessing.py` rather than being added to `loader.py`.

---

## Creating a Feature Branch

Create a separate branch for each feature, bug fix, or improvement:

```bash
git checkout -b feature/short-description
```

Use a descriptive branch name, such as:

```bash
git checkout -b feature/add-gat-model
git checkout -b fix/data-loader
git checkout -b docs/update-installation-guide
```

---

## Making Changes

When working on a contribution:

1. Create a dedicated branch.
2. Make the required changes.
3. Add or update tests where appropriate.
4. Run the complete test suite.
5. Verify that the project still works with the smoke test.
6. Review your changes before committing.

Run the tests with:

```bash
pytest tests/ -v
```

Run the smoke test with:

```bash
python main.py --stage all --nrows 2000
```

---

## Commit Guidelines

Use clear and descriptive commit messages that explain what the commit changes.

Recommended format:

```bash
git commit -m "Add: short description of the change"
```

Examples:

```bash
git commit -m "Add: Graph Attention Network model"
git commit -m "Fix: handle missing values during preprocessing"
git commit -m "Test: add graph construction test cases"
git commit -m "Docs: update dataset setup instructions"
```

Keep commits focused and avoid combining unrelated changes into a single commit.

---

## Push Your Changes

Push your feature branch to your GitHub fork:

```bash
git push origin feature/short-description
```

Then open a Pull Request against the project's `main` branch.

---

## Pull Request Guidelines

When opening a Pull Request, provide a clear description of your contribution.

Please include:

* **What changed?** — Describe the implementation or modification.
* **Why is it needed?** — Explain the problem or motivation.
* **Testing** — Mention the tests and validation performed.
* **Impact** — Explain any relevant performance, compatibility, or behavioral changes.
* **Limitations** — Mention known limitations or trade-offs, if applicable.

A good Pull Request should be focused, well-tested, and easy to review.

---

## Reporting Issues

If you discover a bug or have an idea for an enhancement, please report it through **GitHub Issues**.

When reporting an issue, include as much relevant information as possible:

* A clear description of the problem.
* Expected behavior.
* Actual behavior.
* Steps required to reproduce the issue.
* Whether `--nrows` was used during testing.
* Python version.
* Operating system.
* Relevant error messages or logs.
* Additional context that may help reproduce the problem.

This information helps maintainers investigate and resolve issues more efficiently.

---

## Code of Conduct

This project follows the **Contributor Covenant Code of Conduct**.

By participating in this project, you agree to follow the project's standards for respectful, constructive, and inclusive collaboration.

---

## Contribution Checklist

Before submitting a Pull Request, please confirm that:

* [ ] The project runs successfully in my development environment.
* [ ] I followed the existing project structure and coding style.
* [ ] I added or updated tests where necessary.
* [ ] All tests pass successfully.
* [ ] The smoke test runs successfully.
* [ ] I used a descriptive branch name.
* [ ] My commit messages clearly describe the changes.
* [ ] I documented any important limitations or design decisions.
* [ ] I have reviewed my changes before submitting the Pull Request.

---

## Thank You!

Thank you for contributing to this project. Every improvement, bug report, test, documentation update, and idea helps make this IoT intrusion detection research more robust and useful.
