import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import os

def merge_five_images(image_files, save_file=None):
    fig = plt.figure(figsize=(6,6))
    spec = gridspec.GridSpec(3, 2, figure=fig)

    # First row (two images)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])

    # Second row (two images)
    ax3 = fig.add_subplot(spec[1, 0])
    ax4 = fig.add_subplot(spec[1, 1])

    # Third row (one centered image)
    ax5 = fig.add_subplot(spec[2, :])  # Span across both columns

    axes = [ax1, ax2, ax3, ax4, ax5]

    # Load and display images
    for i, ax in enumerate(axes):
        img = mpimg.imread(image_files[i])
        ax.imshow(img)
        ax.axis('off')  # Hide axes for cleaner look

    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()

def driver(dir, file_prefix, file_suffix, range, save_file=None):
    images = [f'{dir}/{file_prefix}{i}{file_suffix}' for i in range]
    merge_five_images(images, save_file)

if __name__=='__main__':
    merged_plots_dir = 'plots/merged_plots'
    if not os.path.exists(merged_plots_dir):
        os.makedirs(merged_plots_dir)
    
    # Merge plots for HHH k of n case
    hhh_k_n_dir = 'plots/hhh_k_of_n'
    hhh_k_n_pre = 'prob_make_given_k_out_of_'
    hhh_k_n_suf = '_shots.png'
    driver(hhh_k_n_dir, hhh_k_n_pre, hhh_k_n_suf, range(1,6), f'{merged_plots_dir}/hhh_k_of_n_1_to_5.jpg')
    driver(hhh_k_n_dir, hhh_k_n_pre, hhh_k_n_suf, range(6,11), f'{merged_plots_dir}/hhh_k_of_n_6_to_10.jpg')

    # Merge plots for DAHHH DSS case
    dahhh_dss_dir = 'plots/dahhh_DSS'
    dahhh_dss_pre = 'prob_make_given_DSS_of_prev_'
    dahhh_dss_suf = '_shots_line.png'
    driver(dahhh_dss_dir, dahhh_dss_pre, dahhh_dss_suf, range(1,6), f'{merged_plots_dir}/dahhh_dss_1_to_5.jpg')
    driver(dahhh_dss_dir, dahhh_dss_pre, dahhh_dss_suf, range(6,11), f'{merged_plots_dir}/dahhh_dss_6_to_10.jpg')

    # Merge plots for DAHHH DPTS case
    dahhh_dpts_dir = 'plots/dahhh_DPTS'
    dahhh_dpts_pre = 'prob_make_given_DPTS_of_prev_'
    dahhh_dpts_suf = '_shots_line.png'
    driver(dahhh_dpts_dir, dahhh_dpts_pre, dahhh_dpts_suf, range(1,6), f'{merged_plots_dir}/dahhh_dpts_1_to_5.jpg')
    driver(dahhh_dpts_dir, dahhh_dpts_pre, dahhh_dpts_suf, range(6,11), f'{merged_plots_dir}/dahhh_dpts_6_to_10.jpg')