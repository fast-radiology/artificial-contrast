# artificial-contrast
Deep learning for cerebral angiography segmentation from non-contrast computed tomography

## Repository
Requirements: python 3.6+

### Modules
1. `./src/` contains `artificial-contrast` library used later in scripts
2. `./scripts/` contains python scripts needed to reproduce results of our study
3. `./data/` contains sample random data required to run the code.
4. `./examples/` contains example scripts with all environmental variables set


### Installation
1. create new virtual environment
2. install library using command: `$ pip install ./src`

### Data structure
For each step we require one directory `train` - for cross validation and `test` for final test of the best model chosen in cross validation process.
Each patient should have `BC` directory with non-contrast examinations and `label` with corresponding masks.


## Steps to reproduce

1. Generate experiments: `$ ./examples/generate_experiments.sh`. Results can be found in `./results/config/`
2. Run CV (results are always stored by patient and by fold):
    1. `$ ./examples/cv_simple_window.sh` Preprocessing approach: simple, Configuration: Radiodensity range from -100 HU to 300 HU on all 3 channels. Results in:
        - `./results/simple_window_-100_300_result.csv`
        - `./results/simple_window_-100_300_by_patient_result.csv`
    2. `$ ./examples/cv_simple_multiple_windows.sh` Preprocessing approach: simple, Configuration: Radiodensity range: (-40, 120), (-100, 300), (300, 2000) HU. Results in:
        - `./results/simple_multiple_windows_result.csv`
        - `./results/simple_multiple_windows_by_patient_result.csv`
    3. `$ ./examples/cv_uniform_no_limit.sh` Preprocessing approach: Uniform, Configuration: Without clipping radiodensity range. Results in:
        - `./results/freqs_no_limit_window_result.csv`
        - `./results/freqs_no_limit_window_by_patient_result.csv`
    4. `$ ./examples/cv_uniform_window.sh` Preprocessing approach: Uniform, Configuration: Radiodensity range from-100 HU to 300 HU
        - `./results/freqs_window_-100_300_result.csv`
        - `./results/freqs_window_-100_300_by_patient_result.csv`
3. Train model chosen using CV using whole training set and evaluate its performance using test set. (In our case the best model was: Preprocessing approach: simple, Configuration: Radiodensity range: (-40, 120), (-100, 300), (300, 2000) HU.): `$ ./examples/test.sh`. <br> NOTE we used: `export DCM_CONF="{\"windows\": [[-40, 120], [-100, 300], [300, 2000]], \"norm_stats\": [[0.191989466547966, 0.1603623628616333, 0.02605995163321495], [0.3100860118865967, 0.2717258334159851, 0.1233396977186203]]}"` (this is one of parameters created in first step). <br> Final result can be found in: `./results/simple_multiple_windows_testset_result.csv`


## Final model weights

Model weights are available [here](https://storage.cloud.google.com/public-fast-radiology/artificial_contrast_simple_multiple_windows.pth)
