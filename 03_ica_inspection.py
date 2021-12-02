"""Inspect ICA results and apply them.

- Load ICA object
- Using the EOG and ECG channels to automatically mark bad components in the ICA
- manually inspect all components, using automatically marked ones as guidance
    - select additional bad components
    - disselect false positive bad components from automatic marking
- Apply the ICA to the concatenated raw data
    - load the data fresh!
    - this is NOT the data preprocessed for ICA
- preprocess ICA cleaned data with the following steps:
    - bandpass filtering
    - interpolation of bad channels
    - re-referencing to average
- Save the ICA cleaned, preprocessed data


"""
