"""
This script can be used to analyze the logs from senate.
Running the following command will generate such a log file:
    sacct --user=dfrezza --format=jobid,jobname,exitcode,state,avecpu,cputime,cputimeraw,elapsed,avevmsize --start=11/18/21 --end=12/9/21 > logs.txt
"""
import seaborn as sns
import pandas as pd
import re
pd.set_option('display.max_columns', 500)

f1 = open('data/scratch/logs.txt', 'r')
x = f1.readlines()
f1.close()

y = [re.sub(' +', ' ', i).replace(' \n', '') for i in x]

z = []
jobid_map = {}
for i in range(2, len(y)):
    ysplit = y[i].split(' ')
    if ysplit[1] == 'batch':
        z += [ysplit]
    if ysplit[1] not in ['extern', 'batch']:
        jobid_map[ysplit[0].split('_')[0]] = ysplit[1]

df = pd.DataFrame(z)
df.columns = y[0].split(' ')

df['batch_id'] = [i.split('_')[0] for i in df['JobID']]

names = ['tiling', 'nuc_det', 'dtfe', 'clst_det', 'clst_loop']
keep_ids = []
for bid in jobid_map:
    if jobid_map[bid] in names:
        keep_ids += [bid]

df2 = df[(df.batch_id.isin(keep_ids)) & (df.ExitCode == '0:0')].reset_index(drop = True)
df2['JobType'] = [jobid_map[bid] for bid in df2['batch_id']]

div = {'K': 2**20, 'M': 2**10}
memsize = []
for i0 in range(df2.shape[0]):
    avms = df2['AveVMSize'][i0]
    if avms is not None:
        suffix = avms[-1]
        memsize += [float(avms[:-1]) / div[suffix]]
    else:
        memsize += [None]

df2['Memory'] = memsize
df2['Time'] = pd.to_numeric(df2['CPUTimeRAW']) / 60

gbm = df2.groupby(['JobType'])[['Time', 'Memory']].median().reset_index()
gbstd = df2.groupby(['JobType'])[['Time', 'Memory']].std().reset_index()
gball = gbm.merge(gbstd, how = 'inner', on = 'JobType')
gball.columns = ['Job Type', 'Average Time (min)', 'Average Memory (Gb)',
                 'Standard Deviation Time (min)', 'Standard Deviation Memory (Gb)']

sns.boxplot(data = df2, x = 'JobType', y = 'Time', hue = 'JobType')
sns.boxplot(data = df2, x = 'JobType', y = 'Memory', hue = 'JobType')
