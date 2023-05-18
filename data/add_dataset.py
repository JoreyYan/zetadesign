import pickle
import glob
lists=['/home/omnisky/data/everyOne/yjy/pdhs/dataset/S40_train_bfssecen_fix256.pkl','/home/omnisky/data/everyOne/yjy/pdhs/data/dataset/S40_finetunetest_bfssecen_fix256.pkl']
vio=[]

for i in lists:
    pkl_file = open(i, 'rb')
    dataset = pickle.load(pkl_file)
    vio=vio+dataset

output = open('/home/omnisky/data/everyOne/yjy/pdhs/data/dataset/S40_trainfinetunetest_bfssecen_fix256.pkl', 'wb')

# 写入到文件
pickle.dump(vio, output)
output.close()