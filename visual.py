import matplotlib.pyplot as plt
import os
from data.ui_graph import Interaction
def create_directory_if_not_exists(path):
    # 检查路径是否存在
    if not os.path.exists(path):
        # 如果路径不存在，递归创建路径
        os.makedirs(path)
        print(f"Directory '{path}' created.")

# 使用示例
# create_directory_if_not_exists('/path/to/your/directory')

def plt_loss(rec_loss_list , cl_loss_list , batch_loss_list , epoch , filename) :

    # 绘制损失曲线
    fig , ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.plot(rec_loss_list, 'r-' , label='Reconstruction Loss')
    plt.plot(cl_loss_list, 'c-' , label='Contrastive Loss')
    plt.plot(batch_loss_list ,'b-', label='Batch Loss')

    ax.set_title(f'Loss Over {epoch}')
    ax.set_xlabel(f'{epoch}')
    ax.set_ylabel('Loss')

    plt.legend()
    # 保存图像到文件
    # plt.savefig(f'results/log_weightInfoNCE_100_epochs_2_layer/all_epoch.png')
    create_directory_if_not_exists(f'results/{filename}')
    plt.savefig(f'results/{filename}/{epoch}.png')
    plt.close('all')  # 关闭所有图形

from data.loader import FileIO
import os
from util.conf import ModelConf
import torch
from base.torch_interface import TorchGraphInterface
import scipy.sparse as sp

def visual_htop(dataset) :
    dataset_path = os.path.dirname(__file__)
    dataset_path = os.path.join(dataset_path , f'dataset/{dataset}')
    training_data = FileIO.load_data_set(f'{dataset_path}/train.txt', 'graph')
    test_data = FileIO.load_data_set(f'{dataset_path}/test.txt', 'graph')
    data = Interaction(ModelConf('./conf/SimGCL.conf') , training=training_data , test=test_data)
    ui_adj = data.interaction_mat
    import pdb
    pdb.set_trace()
    ui_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(ui_adj).cuda()
    two_htop_uu = torch.sparse.mm(ui_adj , ui_adj.t())
    three_htop_ui = torch.sparse.mm(two_htop_uu , ui_adj)
    
    htop = {}
    tt = ui_adj.tocoo()
    deg = tt.sum(1)
    rows , cols = tt.row , tt.col
    for i in range(len(rows)) :
        htop[(rows[i] , cols[i])] = 1
    

    tt = three_htop_ui.tocoo()
    rows , cols = tt.row , tt.col
    for i in range(len(rows)) :
        if (rows[i] , cols[i]) not in htop :
            htop[(rows[i] , cols[i])] = 3


    result = {}
    for t in data.test_data :
        if (t[0] , t[1]) in htop :
            result[(deg[t[0]] , htop[(t[0] , t[1])])] += 1
        else :
            result[(deg[t[0]] , 5)] += 1

    # 提取键和值
    keys = list(result.keys())
    values = list(result.values())

    # 创建条形图
    plt.figure(figsize=(8, 4))  # 设置图形的大小
    plt.bar(keys, values, color='skyblue')  # 创建条形图

    # 添加标题和轴标签
    plt.title('Visualization of Dictionary Result')
    plt.xlabel('Key')
    plt.ylabel('Value')

    # 显示图形
    plt.savefig('test_visual.png')

# visual_htop('yelp2018')

def visitRecall():
    files = ['log_FusionAttentionGNN_0.05.txt' , 'log_FusionAttentionGNN_0.1.txt']
    lambdas = [0.05 , 0.1]
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    epochs = 60
    for i in range(len(files)):
        recalls = []
        ndcgs = []
        with open(files[i] , 'r') as f :
            txt = f.read()
            # import pdb
            # pdb.set_trace()
            res = [t for t in txt.split('\n') if 'Epoch' in t[:10]][::2]
            for j in range(min(epochs,len(res))):
                t = res[j].replace(' ' , '').replace('|',',').split(',')
                recall_20 = float(t[3].split(':')[1])
                ndcg_20 = float(t[4].split(':')[1])
                recalls.append(recall_20)
                ndcgs.append(ndcg_20)

        axs[0].plot(range(0,min(epochs,len(recalls))) , recalls , label=f'cl_rate = {lambdas[i]}')
        axs[1].plot(range(0,min(epochs,len(ndcgs))) , ndcgs , label=f'cl_rate = {lambdas[i]}')
        axs[0].set_title('Recall@20/epoch')  # 设置标题
        axs[0].set_xlabel('epoch')  # 设置X轴标签
        axs[0].set_ylabel('Recall@20')  # 设置Y轴标签
        axs[1].set_title('NDCG@20/epoch')  # 设置标题
        axs[1].set_xlabel('epoch')  # 设置X轴标签
        axs[1].set_ylabel('NDCG@20')  # 设置Y轴标签
        axs[0].legend()
        axs[1].legend()
    plt.tight_layout()
    plt.legend()
    plt.savefig('FusionAttentionGNN')

visitRecall()
