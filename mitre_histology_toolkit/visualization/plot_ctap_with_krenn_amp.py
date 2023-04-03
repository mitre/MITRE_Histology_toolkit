import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_csv('data/external/AMP_CTAP_donor_mapping_nuclei_density_dtfe.csv')

clinical = pd.read_excel('data/external/RARecruitment_20200323_AF_sorted105_clin_3grades_BAAF.xlsx', engine = 'openpyxl')

cdata = clinical[['Subject\nID', 'Krenn_inflam_avg']]
cdata.columns = ['subject_id', 'krenn_avg']

comb = data.merge(cdata)

krenn_avg = []
for i in comb.krenn_avg:
    if i == 'na':
        krenn_avg += [np.nan]
    else:
        krenn_avg += [float(i)]

comb['krenn_avg'] = krenn_avg
comb['krenn_avg'] = comb['krenn_avg'].round(0)

sns.boxplot(x = 'krenn_avg', y = 'avg_dtfe', hue = 'krenn_avg',
            data = comb[comb.krenn_avg.isnull() == False], palette = 'Dark2')
plt.show()
sns.boxplot(x = 'krenn_avg', y = 'avg_nuclei_density', hue = 'krenn_avg',
            data = comb[comb.krenn_avg.isnull() == False], palette = 'Dark2')
plt.show()
