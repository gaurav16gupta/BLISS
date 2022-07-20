# creates all folders etc
from utils import create_tfrecords_ann, label_tfrecords, create_universal_lookups, create_query_lookups
import glob
import time
from multiprocessing import Pool
from config import train_config as config

######## Create TF Records ##########
begin_time = time.time()
files = glob.glob(config.train_data_loc+'*.txt')
for file in files:
    nothing = create_tfrecords_ann(file)

print('elapsed_time:', time.time()-begin_time)

######## TF Records for label titles ##########
begin_time = time.time()
files = glob.glob(config.train_data_loc+'docs.txt')
for file in files:
    nothing = label_tfrecords(file)

print('elapsed_time:', time.time()-begin_time)

########## Prepare Label lookups (for MACH grouping)
begin_time = time.time()
p = Pool(16)
p.map(create_universal_lookups, list(range(16)))
p.close()
p.join()
print('elapsed_time:', time.time()-begin_time)

# ########## Prepare input idx lookups (for feature hashing)
begin_time = time.time()
p = Pool(16)
p.map(create_query_lookups, list(range(16)))
p.close()
p.join()
print('elapsed_time:', time.time()-begin_time)

