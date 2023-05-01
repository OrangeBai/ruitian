from utils.pre import check_date_sanity, write_feat_uid, pre_process, compute_std, compute_mean
import yaml

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.full_load(f)

    base_dir = config['dataset']['base']['base_dir']
    proc_dir = config['dataset']['base']['proc_dir']
    check_date_sanity(base_dir)
    write_feat_uid(base_dir)
    pre_process(base_dir, proc_dir)
    compute_mean(base_dir, proc_dir)
    compute_std(base_dir, proc_dir)
