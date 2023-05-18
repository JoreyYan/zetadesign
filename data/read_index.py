import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
def aa_psy_che(indexfile='/CLIP/data/AAindex1',lookup = 'XACDEFGHIKLMNPQRSTVWY'):

    title='A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V'
    oldindex=title.split()
    index=[]
    for i in oldindex:
        index.append(i[0])
    for i in oldindex:
        index.append(i[2])


    with open(indexfile) as f:
        contents = f.readlines()
    n=0
    data=[]
    for i in range(len(contents)):
        this_proper=[]

        if contents[i][:1]=='I':
            n=n+1
            for a in list(contents[i + 1].split('\n')[0].split()):
                if a == 'NA':
                    a = 0.
                this_proper.append(float(a) )
            for a in list(contents[i + 2].split('\n')[0].split()):
                if a=='NA':
                    a=0.

                this_proper.append(float(a))

            data.append(this_proper)

    npdata=torch.asarray(data)
    npx=torch.mean(npdata,-1).unsqueeze(-1)
    digits=torch.transpose(torch.cat((npdata,npx),-1),0,1)
    index.append('X')

    new_list=[]
    for i in list(lookup):
        aa=index.index(i)
        new_list.append(digits[aa])

    digits=torch.stack((new_list),0)
    return digits

def show():
    # 创建一个随机的PCA模型，该模型包含两个组件
    amino_acids_Weight = {}
    amino_acids_Weight["A"] = {"name": "Alanine", "weight": 89.1}
    amino_acids_Weight["R"] = {"name": "Arginine", "weight": 174.2}
    amino_acids_Weight["N"] = {"name": "Asparigine", "weight": 132.1}
    amino_acids_Weight["D"] = {"name": "Aspartate", "weight": 133.1}
    amino_acids_Weight["C"] = {"name": "Cysteine", "weight": 121.2}
    amino_acids_Weight["E"] = {"name": "Glutamate", "weight": 147.1}
    amino_acids_Weight["Q"] = {"name": "Glutamine", "weight": 146.2}
    amino_acids_Weight["G"] = {"name": "Glycine", "weight": 75.1}
    amino_acids_Weight["H"] = {"name": "Histidine", "weight": 155.2}
    amino_acids_Weight["I"] = {"name": "Isoleucine", "weight": 131.2}
    amino_acids_Weight["L"] = {"name": "Leucine", "weight": 131.2}
    amino_acids_Weight["K"] = {"name": "Lysine", "weight": 146.2}
    amino_acids_Weight["M"] = {"name": "Methionine", "weight": 149.2}
    amino_acids_Weight["F"] = {"name": "Phenylalanine", "weight": 165.2}
    amino_acids_Weight["P"] = {"name": "Proline", "weight": 115.1}
    amino_acids_Weight["S"] = {"name": "Serine", "weight": 105.1}
    amino_acids_Weight["T"] = {"name": "Threonine", "weight": 119.1}
    amino_acids_Weight["W"] = {"name": "Tryptophan", "weight": 204.2}
    amino_acids_Weight["Y"] = {"name": "Tyrosine", "weight": 181.2}
    amino_acids_Weight["V"] = {"name": "Valine", "weight": 117.1}
    amino_acids_Weight["X"] = {"name": "X", "weight": 136.90}



    randomized_pca = PCA(n_components=2, svd_solver='randomized')
    lookup='XACDEFGHIKLMNPQRSTVWY'
    # digits=aa_psy_che(indexfile='AAindex1')
    digits=torch.load('570aaf.pth')
    # digits=digits[:,:4]
    digits=F.normalize(digits,p=2,dim=0)

    digits=np.asarray(digits)
    # 拟合数据并将其转换为模型
    reduced_data_rpca = randomized_pca.fit_transform(digits.data)
    # 创建一个常规的PCA模型
    pca = PCA(n_components=2)
    # 拟合数据并将其转换为模型
    reduced_data_pca = pca.fit_transform(digits.data)

    clo=np.arange(len(lookup))
    #for i in range(len(lookup)):
    x = reduced_data_rpca[:, 0]
    y = reduced_data_rpca[:, 1]
    fig,ax = plt.subplots(figsize=(10,10))
    txt=list(lookup)
    scatter=plt.scatter(x, y,c=clo,label=list(lookup))
    for i in range(len(lookup)):
        ax.annotate(amino_acids_Weight[lookup[i]]['name'], (x[i], y[i]), xytext=(10,10), textcoords='offset points')
    plt.legend(*scatter.legend_elements(),title="classes")
    # 设置图例，0-9用不同颜色表示
    plt.legend(clo, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # 设置坐标标签
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    # 设置标题
    plt.title("PCA Scatter Plot")

    # 显示图形
    plt.show()

if __name__ == '__main__':
    show()