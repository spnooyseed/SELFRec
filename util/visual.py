import matplotlib.pyplot as plt
import os

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