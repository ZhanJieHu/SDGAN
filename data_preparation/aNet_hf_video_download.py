# 下载huggingface上指定数据集的指定文件，到指定位置。
def set_proxy():
    import os
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    os.environ["all_proxy"] = "http://127.0.0.1:7890"


def huggingface_download(repo_id, dataset_dir, filename):
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",        # 明确指定这是数据集类型
        filename=filename,
        local_dir=dataset_dir,
        local_dir_use_symlinks=False,  # 很重要，建议加上
    )

    print("Downloaded to:", local_path)

files = [
"missing_files_v1-3_test.zip",
"README.md",
"activity_net.v1-3.min.json",
"missing_files.zip",
"missing_files_v1-2_test.zip",
"missing_files_v1-3_test.zip",
"train.json",
"train_ids.json",
"v1-2_test.tar.gz.00",
"v1-2_test.tar.gz.01",
"v1-2_test.tar.gz.02",
"v1-2_test.tar.gz.03",
"v1-2_test.tar.gz.04",
"v1-2_test.tar.gz.05",
"v1-2_test.tar.gz.06",
"v1-2_test.tar.gz.07",
"v1-2_test.tar.gz.08",
"v1-2_test.tar.gz.09",
"v1-2_test.tar.gz.10",
"v1-2_test.tar.gz.11",
"v1-2_test.tar.gz.12",
"v1-2_test.tar.gz.13",
"v1-2_train.tar.gz.00",
"v1-2_train.tar.gz.01",
"v1-2_train.tar.gz.02",
"v1-2_train.tar.gz.03",
"v1-2_train.tar.gz.04",
"v1-2_train.tar.gz.05",
"v1-2_train.tar.gz.06",
"v1-2_train.tar.gz.07",
"v1-2_train.tar.gz.08",
"v1-2_train.tar.gz.09",
"v1-2_train.tar.gz.10",
"v1-2_train.tar.gz.11",
"v1-2_train.tar.gz.12",
"v1-2_train.tar.gz.13",
"v1-2_train.tar.gz.14",
"v1-2_train.tar.gz.15",
"v1-2_train.tar.gz.16",
"v1-2_train.tar.gz.17",
"v1-2_train.tar.gz.18",
"v1-2_train.tar.gz.19",
"v1-2_train.tar.gz.20",
"v1-2_train.tar.gz.21",
"v1-2_train.tar.gz.22",
"v1-2_train.tar.gz.23",
"v1-2_train.tar.gz.24",
"v1-2_train.tar.gz.25",
"v1-2_train.tar.gz.26",
"v1-2_train.tar.gz.27",
"v1-2_train.tar.gz.28",
"v1-2_val.tar.gz.00",
"v1-2_val.tar.gz.01",
"v1-2_val.tar.gz.02",
"v1-2_val.tar.gz.03",
"v1-2_val.tar.gz.04",
"v1-2_val.tar.gz.05",
"v1-2_val.tar.gz.06",
"v1-2_val.tar.gz.07",
"v1-2_val.tar.gz.08",
"v1-2_val.tar.gz.09",
"v1-2_val.tar.gz.10",
"v1-2_val.tar.gz.11",
"v1-2_val.tar.gz.12",
"v1-2_val.tar.gz.13",
"val_1.json",
"val_ids.json",
"v1-3/train_val/v_0gLAhptj34w.mp4",
"v1-3/train_val/v_rZmNsUX-7SU.mp4",
"v1-3/train_val/v_y80Jbcb5GWA.mp4"
]

if __name__ == "__main__":
    repo_id = "YimuWang/ActivityNet"   # Hugging Face 上的数据集 repo              
    dataset_dir = "./datasets/activitynet/videos"
    set_proxy()

    for filename in files:
        huggingface_download(repo_id, dataset_dir, filename)



