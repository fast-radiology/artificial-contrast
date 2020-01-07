import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.stats.api as sms

from artificial_contrast.const import SIMPLE_MULTIPLE_WINDOWS
from artificial_contrast.data import get_scans, get_patients
from artificial_contrast.dicom import get_patient_sex


DATA_PATH = os.environ['DATA']
data_path = Path(DATA_PATH)
patients = get_patients(data_path)
scans = get_scans(data_path)

gender_results = []
for patient in patients:
    for scan in scans:
        if patient in scan:
            gender_results.append({'patient': patient, 'gender': get_patient_sex(scan)})
            break

gender_df = pd.DataFrame(gender_results)

df = pd.read_csv(f"{SIMPLE_MULTIPLE_WINDOWS}_by_patient_result.csv").merge(gender_df)

print(df.gender.value_counts())

male_df = df[df.gender == 'M']
female_df = df[df.gender == 'F']

plt.figure()
male_df.dice.hist(bins=20)
plt.savefig('male.png')

plt.figure()
female_df.dice.hist(bins=20)
plt.savefig('female.png')

print(f"Male var: {male_df.dice.var()}; Female var: {female_df.dice.var()}")

print('t-test (tstat, pvalue, df)')
print(sms.ttest_ind(male_df.dice.values, female_df.dice.values))

cm = sms.CompareMeans(
    sms.DescrStatsW(male_df.dice.values), sms.DescrStatsW(female_df.dice.values)
)
print(f"CI: {cm.tconfint_diff(usevar='unequal')}")
