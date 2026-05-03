from UTiLs import pobj, save_file
from .node2list import nclip2poolinglist
from UTiLs import _update_cfg_nodePro
import os
from yacs.config import CfgNode as CN
import yaml
import errno
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def create_cfg_from_yaml(yaml_file):
    """通过YAML文件创建配置"""
    # 先读取YAML文件内容
    with open(yaml_file, 'r') as f:
        yaml_content = yaml.safe_load(f)
    
    # 创建空的CfgNode
    cfg = CN()
    
    # 递归构建配置结构
    def build_cfg_from_dict(d, parent_node):
        for key, value in d.items():
            if isinstance(value, dict):
                # 如果是字典，创建新的CN节点
                parent_node[key] = CN()
                build_cfg_from_dict(value, parent_node[key])
            else:
                # 直接设置值
                parent_node[key] = value
    
    build_cfg_from_dict(yaml_content, cfg)
    return cfg

def read_config(config_path, args):
    global cfg
    # 直接从YAML创建配置
    cfg = create_cfg_from_yaml(config_path)
    cfg = setting_training_stages(cfg)

    # 合并命令行参数
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    cfg.MODEL.DEVICE = args.device
    print(f"Using GPU: {cfg.MODEL.DEVICE}")
    cfg.freeze()

def check_feat_len(cfg):
    # 根据代码逻辑检查cfg中有没有互相抵触的地方
    if(cfg.DATASETS.nclip_in_v_feat != cfg.MODEL.video.encoder.v_feat_len):
        assert False, "两处video_feat长度不同！"
    
    if(cfg.DATASETS.f_feat_dim != cfg.MODEL.frame.encoder.f_feat_dim):
        assert False, "两处frame_feat维度不同！"


from .checkPoolingList import validate_pooling_list
def check_twostages(cfg):
    # two stage checking (checking pooling list indeed)
    NUM_CLIPS = cfg.MODEL.NUM_CLIPS
    merge_num = cfg.MODEL.merge_num
    assert NUM_CLIPS % merge_num == 0, "Ensure that the number of nodes is divisible"

    video_pooling_list = cfg.MODEL.video.temporal_feature_map.pooling_list
    validate_pooling_list(video_pooling_list, NUM_CLIPS)
    
    frame_pooling_list = cfg.MODEL.frame.temporal_feature_map.pooling_list
    validate_pooling_list(frame_pooling_list, NUM_CLIPS)

def mask_pooling_list(cfg):
    # two stage checking (checking pooling list indeed)
    NUM_CLIPS = cfg.MODEL.NUM_CLIPS
    merge_num = cfg.MODEL.merge_num
    assert NUM_CLIPS % merge_num == 0, "Ensure that the number of nodes is divisible"

    def mask_check(mask_config, n_clips):
        if mask_config['mode'] == "sparse":
            pooling_list = mask_config['pooling_list']
            validate_pooling_list(pooling_list, n_clips)
    
    mask_check(cfg.MODEL.video.mask_1, n_clips = NUM_CLIPS // merge_num)
    mask_check(cfg.MODEL.video.mask_2, n_clips = NUM_CLIPS)
    mask_check(cfg.MODEL.frame.mask_1, n_clips = NUM_CLIPS // merge_num)
    mask_check(cfg.MODEL.frame.mask_2, n_clips = NUM_CLIPS)

class ConfigManager:
    # 定义可用的验证函数映射表
    VALIDATION_FUNCTIONS = {
        "check_feat_len": check_feat_len,
        "check_twostages": check_twostages,
        "mask_pooling_list": mask_pooling_list,
    }

    @staticmethod
    def setting_MAX_EPOCH(cfg):
        """
        在cfg['SOLVER']中添加MAX_EPOCH配置项
        
        参数:
            cfg: 配置字典
            
        返回:
            更新后的cfg字典
        """
        if 'MAX_EPOCH' in cfg['SOLVER']:
            return cfg

        # 计算训练阶段的总epoch数
        total_epochs = 0
        
        # 检查是否存在训练阶段配置
        assert 'training_stages' in cfg['SOLVER'], "SOLVER配置中未找到training_stages"

        for stage in cfg['SOLVER']['training_stages']:
            if 'NumberOfEpochs' in stage:
                total_epochs += stage['NumberOfEpochs']
        
        # 添加MAX_EPOCH到SOLVER配置中
        cfg['SOLVER']['MAX_EPOCH'] = total_epochs
        
        print(f"已添加 MAX_EPOCH = {total_epochs} 到 SOLVER 配置中")

        return cfg

    """统一管理配置的加载、验证和保存"""
    @staticmethod
    def load_config(yaml_path: str, args) -> CN:
        """加载并验证配置"""
        cfg = create_cfg_from_yaml(yaml_path)
        cfg = ConfigManager.setting_MAX_EPOCH(cfg)
        
        if args.opts:
            cfg.merge_from_list(args.opts)
        
        cfg.MODEL.DEVICE = args.device
        cfg.freeze()
        return cfg
    
    @staticmethod
    def setup_output_dir(cfg: CN, tag: str = None, resume_dir: str = None) -> CN:
        """设置输出目录"""
        cfg = cfg.clone()
        cfg.defrost()
        
        if resume_dir:
            cfg.OUTPUT_DIR = resume_dir
        elif tag:
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, tag)
        
        cfg.freeze()
        mkdir(cfg.OUTPUT_DIR)
        return cfg

    @staticmethod
    def set_resume_mode(cfg: CN, resume: bool) -> CN:
        """设置是否为恢复训练模式"""
        cfg = cfg.clone()
        cfg.defrost()
        cfg.SOLVER.RESUME = resume
        cfg.freeze()
        return cfg
    
    @staticmethod
    def setup_logging(cfg, args):
        from model_all.utils.comm import get_rank
        from model_all.utils.logger import setup_logger_time
        

        logger = setup_logger_time("dtf", cfg.OUTPUT_DIR, get_rank()) 
        logger.info("Using {} GPUs".format(args.num_gpus))
        logger.info("Loaded args {}".format(args))
        logger.info("args.config_file: {}".format(args.config_file))
        output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
        logger.info("Save config into: {}".format(output_config_path))

    @staticmethod
    def set_checkpoint_dir(cfg: CN, checkpoint_dir: str = None, tag=None) -> CN:
        """设置输出目录"""
        if tag == None: # checkpoint_dir is guaranteed to be right
            assert os.path.exists(checkpoint_dir), f"路径 {checkpoint_dir} 不存在"
            cfg = cfg.clone()
            cfg.defrost()    
            cfg.OUTPUT_DIR = checkpoint_dir    
            cfg.freeze()
        else:
            # checkpoint_dir is floder stores all checkpoint
            # tag is the name of this checkpoint
            assert checkpoint_dir.endswith('/'), "OUTPUT_DIR must end with '/'"
            cfg.defrost()
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, tag)
            cfg.freeze()  # 重新冻结（可选）
        return cfg
        
    @staticmethod
    def validate_config(cfg):
        """统一管理配置的加载、验证和保存"""
        # 从配置中获取要执行的验证函数列表
        funList = cfg.get("validation_functions", [])
        
        if not funList:
            print("警告: 未指定验证函数，跳过配置验证")
            return
        
        print(f"开始配置验证，共 {len(funList)} 个检查项...")
        
        for fun_name in funList:
            if fun_name not in ConfigManager.VALIDATION_FUNCTIONS:
                raise ValueError(f"未知的验证函数: {fun_name}")
            
            try:
                print(f"  执行检查: {fun_name}")
                validation_func = ConfigManager.VALIDATION_FUNCTIONS[fun_name]
                validation_func(cfg)
                print(f"  ✓ {fun_name} 通过")
            except AssertionError as e:
                print(f"  ✗ {fun_name} 失败: {e}")
                raise
            except Exception as e:
                print(f"  ✗ {fun_name} 出错: {e}")
                raise
        
        print("配置验证完成！")

    # 处理 video 和 frame 的通用逻辑
    @staticmethod
    def process_feature_cfg(feature_cfg, hidden_dim, Nclip_inV, merge_num=1):
        from UTiLs import _update_cfg_nodePro
        """处理 video 或 frame 配置的通用函数"""
        updated_configs = {}
        
        # Encoder
        encoder_config_raw = feature_cfg.get("encoder")
        if encoder_config_raw is not None:
            updated_configs["encoder"] = _update_cfg_nodePro(
                encoder_config_raw,
                hidden_dim=hidden_dim,
                Nclip_inV=Nclip_inV
            )


        mask_1_config_raw = feature_cfg.get("mask_1")
        if mask_1_config_raw is not None:
            updated_configs["mask_1"] = _update_cfg_nodePro( 
            cfg_node=mask_1_config_raw,
            Nclip_inV=Nclip_inV // merge_num)

            pooling_list_first = mask_1_config_raw.get("pooling_list")
            if pooling_list_first is None:
                pooling_list_first = [(Nclip_inV // merge_num) - 1]
        else:
            raise ValueError("fail to: feature_cfg.get(\"mask_1\")")
        
        mask_2_config_raw = feature_cfg.get("mask_2")
        if mask_2_config_raw is not None:
            assert merge_num > 1,  "Get mask_2 in config. But merge_num = {merge_num}, "
            "which means second stage is not support."
            updated_configs["mask_2"] = _update_cfg_nodePro( 
            cfg_node=mask_2_config_raw,
            Nclip_inV=Nclip_inV)

            pooling_list_second = mask_2_config_raw.get("pooling_list")
            if pooling_list_second is None:
                pooling_list_second = [Nclip_inV - 1]
        else:
            raise ValueError("fail to: feature_cfg.get(\"mask_2\")")


        # Temporal Feature Map
        tfm_config_raw = feature_cfg.get("temporal_feature_map") 
        if tfm_config_raw is not None:
            # First temporal_feature_map (使用合并后的clip数量)
            Nclip_merged = Nclip_inV // merge_num  # 128 // 8 = 16
            pooling_list_first = pooling_list_first
            updated_configs["temporal_feature_map"] = _update_cfg_nodePro(
                tfm_config_raw,
                hidden_dim=hidden_dim,
                Nclip_inV=Nclip_merged,
                pooling_list=pooling_list_first
            )


            # Second temporal_feature_map (使用merge_num)

            pooling_list_second =  pooling_list_second
            updated_configs["temporal_feature_map_second"] = _update_cfg_nodePro(
                tfm_config_raw,
                hidden_dim=hidden_dim,
                Nclip_inV=Nclip_inV,
                pooling_list=pooling_list_second
            )

        # Proposal Conv
        proposal_config_raw = feature_cfg.get("proposal_conv")
        if proposal_config_raw is not None:
            updated_configs["proposal_conv"] = _update_cfg_nodePro( # first proposal_conv
                proposal_config_raw,
                input_size=hidden_dim,
                output_size=hidden_dim
            )

            updated_configs["proposal_conv_second"] = _update_cfg_nodePro( # second proposal_conv
                proposal_config_raw,
                input_size=hidden_dim,
                output_size=hidden_dim
            )


            
        # if (tfm_config_raw is not None) and (merge_num > 1): # (merge_num > 1) suggests TwoStage
        #     # 321p_feat_generator: generate proposal feature in second stage by turn 3 feature into 1.
        #     updated_configs["321p_feat_generator"] = _update_cfg_nodePro( 
        #         cfg_node=None,
        #         mode = "CustomBERT",
        #         hidden_dim=hidden_dim,
        #         input_size=3,           
        #         output_size=1
        #     )

        # p321_feat_generator_config = feature_cfg.get("p321_feat_generator")
        # if p321_feat_generator_config is not None:
        #     updated_configs["p321_feat_generator"] = _update_cfg_nodePro(
        #         cfg_node=p321_feat_generator_config,
        #         hidden_dim=hidden_dim,
        #         input_size=3,           
        #         output_size=1
        #     )

        # GNN
        gnn_config_raw = feature_cfg.get("gnn")
        if gnn_config_raw is not None:
            updated_configs["gnn"] = _update_cfg_nodePro(
                gnn_config_raw, 
                n_filters=hidden_dim
            )


        
        return updated_configs

    @staticmethod
    def set_mask(n_clip, mask_config):

        # 设置pooling_list
        if mask_config['mode'] == "sparse":
            pooling_list = nclip2poolinglist(n_clip)
        elif mask_config['mode'] == "upper":
            pooling_list = [n_clip - 1]

        # 更新节点数
        updated_config = _update_cfg_nodePro( 
            cfg_node=mask_config,
            Nclip_inV=n_clip,
            pooling_list = pooling_list)
        return updated_config, pooling_list

    # 处理 video 和 frame 的通用逻辑
    @staticmethod
    def process_feature_cfgPlus(feature_cfg, hidden_dim, Nclip_inV, merge_num=1):
        '''升级了设置pooling_list的逻辑'''
        
        """处理 video 或 frame 配置的通用函数"""
        updated_configs = {}
        
        # Encoder
        encoder_config_raw = feature_cfg.get("encoder")
        if encoder_config_raw is not None:
            updated_configs["encoder"] = _update_cfg_nodePro(
                encoder_config_raw,
                hidden_dim=hidden_dim,
                Nclip_inV=Nclip_inV
            )


        mask_1_config_raw = feature_cfg.get("mask_1")
        if mask_1_config_raw is not None:
            updated_configs["mask_1"], pooling_list_first = ConfigManager.set_mask(
                n_clip = Nclip_inV // merge_num, # merge_num=1说明不使用粗细粒度Proposal，mask_1就是Proposal的mask
                mask_config = mask_1_config_raw
            )
        else:
            raise ValueError("fail to: feature_cfg.get(\"mask_1\")")
        
        mask_2_config_raw = feature_cfg.get("mask_2")
        if mask_2_config_raw is not None:
            assert merge_num > 1,  "Get mask_2 in config. But merge_num = {merge_num}, "
            "which means second stage is not support."

            updated_configs["mask_2"], pooling_list_second = ConfigManager.set_mask(
                n_clip = Nclip_inV,
                mask_config = mask_2_config_raw
            )
    
        else:
            raise ValueError("fail to: feature_cfg.get(\"mask_2\")")


        # Temporal Feature Map
        tfm_config_raw = feature_cfg.get("temporal_feature_map") 
        if tfm_config_raw is not None:
            # First temporal_feature_map (使用合并后的clip数量)
            Nclip_merged = Nclip_inV // merge_num  # 128 // 8 = 16
            pooling_list_first = pooling_list_first
            updated_configs["temporal_feature_map"] = _update_cfg_nodePro(
                tfm_config_raw,
                hidden_dim=hidden_dim,
                Nclip_inV=Nclip_merged,
                pooling_list=pooling_list_first
            )


            # Second temporal_feature_map (使用merge_num)

            pooling_list_second =  pooling_list_second
            updated_configs["temporal_feature_map_second"] = _update_cfg_nodePro(
                tfm_config_raw,
                hidden_dim=hidden_dim,
                Nclip_inV=Nclip_inV,
                pooling_list=pooling_list_second
            )

        # Proposal Conv
        proposal_config_raw = feature_cfg.get("proposal_conv")
        if proposal_config_raw is not None:
            updated_configs["proposal_conv"] = _update_cfg_nodePro( # first proposal_conv
                proposal_config_raw,
                input_size=hidden_dim,
                output_size=hidden_dim
            )

            updated_configs["proposal_conv_second"] = _update_cfg_nodePro( # second proposal_conv
                proposal_config_raw,
                input_size=hidden_dim,
                output_size=hidden_dim
            )

        # GNN
        gnn_config_raw = feature_cfg.get("gnn")
        if gnn_config_raw is not None:
            updated_configs["gnn"] = _update_cfg_nodePro(
                gnn_config_raw, 
                n_filters=hidden_dim
            )


        
        return updated_configs

    @staticmethod
    def expand_config(cfg):
        from UTiLs import _update_cfg_nodePro

        Nclip_inV = cfg.MODEL.NUM_CLIPS
        hidden_dim = cfg.MODEL.hidden_dimension
    
        # Text 相关配置
        text_encoder_config = _update_cfg_nodePro(
            cfg.MODEL.text.encoder,
            hidden_dim=hidden_dim,
            dataset_name=cfg.DATASETS.NAME
        )

        text_decoder_config = _update_cfg_nodePro(
            cfg.MODEL.text.decoder,
            hidden_dim=hidden_dim,
            dataset_name=cfg.DATASETS.NAME
        )

        # 位置编码
        pos_config = _update_cfg_nodePro(
            cfg.MODEL.pos_encoding, 
            hidden_dim=hidden_dim
        )

        # Feature Merger
        feature_merger_config = _update_cfg_nodePro(
            cfg.MODEL.feature_merger,
            hidden_dim=hidden_dim,
            merge_num=cfg.MODEL.merge_num,
            NUM_CLIPS=Nclip_inV
        )

        # 更新 MODEL 相关
        cfg.defrost()
        cfg.MODEL.text.encoder = text_encoder_config
        cfg.MODEL.text.decoder = text_decoder_config
        cfg.MODEL.pos_encoding = pos_config

        cfg.MODEL.video.feature_merger = feature_merger_config  # Guarantee that the feature merger is identical to all movie classes.
        cfg.MODEL.frame.feature_merger = feature_merger_config  
        cfg.freeze()    


        # 处理 video 和 frame
        video_updates = ConfigManager.process_feature_cfgPlus(cfg.MODEL.video, hidden_dim, Nclip_inV, merge_num=cfg.MODEL.merge_num)
        frame_updates = ConfigManager.process_feature_cfgPlus(cfg.MODEL.frame, hidden_dim, Nclip_inV, merge_num=cfg.MODEL.merge_num)

        # 更新movie类
        cfg.defrost()
        # 更新 video 相关
        for key, value in video_updates.items():
            setattr(cfg.MODEL.video, key, value)
        
        # 更新 frame 相关
        for key, value in frame_updates.items():
            setattr(cfg.MODEL.frame, key, value)
        cfg.freeze()

        return cfg
    
    @staticmethod
    def save_source_files(output_dir: str, config_file: str):
        """
        保存关键源代码文件
        
        Args:
            output_dir: 输出目录
            config_file: 原始配置文件路径
        """
        files_to_save = {
            config_file: "config_origin.yaml",
            "./model_all/modeling/main_model.py": "main_model.py",
            "./train1.py": "train1.py",
            "./model_all/engine/trainer.py": "trainer.py",
            "./model_all/engine/StageManager.py": "StageManager.py",  # 修正了路径
            "model_all/subassembly/TwoStage/TwoStageManager.py": "TwoStageManager.py",
            "UTiLs/ConfigManager.py": "ConfigManager.py"
        }
        
        for src_file, dst_name in files_to_save.items():
            if os.path.exists(src_file):
                save_file(
                    file=src_file,
                    output_path=output_dir,
                    fileName=dst_name
                )

    @staticmethod
    def save_source_filesPro(output_dir: str, train_file: str, config_file:str):
        '''
        save_source_filesPro 的 Docstring
        
        :param output_dir: 
        :param train_file: train.py path:= project_path/train.py
        :param config_file: 
        '''
        import shutil
        project_path = os.path.dirname(train_file) # 当前文件所在目录
        output_dir = os.path.join(output_dir, "source_project")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir) # rm -rf [output_dir]

        # 复制整个目录树(相当于 cp -r)
        shutil.copytree(os.path.join(project_path, "model_all"), os.path.join(output_dir, "model_all")) # cp -r /media/zhangbolin/hu/OSGs/unet/model_all [output_dir]
        shutil.copytree(os.path.join(project_path, "UTiLs"), os.path.join(output_dir, "UTiLs"))     # cp -r /media/zhangbolin/hu/OSGs/unet/UTiLs [output_dir]
        shutil.copy2(config_file, output_dir)      # cp -r /media/zhangbolin/hu/OSGs/unet/config_file.yaml [output_dir]
        shutil.copy2(train_file, output_dir)                              # cp -r /media/zhangbolin/hu/OSGs/unet/train1.py [output_dir]
        if os.path.exists(os.path.join(project_path, "test1.py")):
            shutil.copy2(os.path.join(project_path, "test1.py"), output_dir)# cp -r /media/zhangbolin/hu/OSGs/unet/test1.py [output_dir]
        
        from UTiLs import write_txt
        note = f"""
        Modifying `sys.path.append('/media/zhangbolin/hu/OSGs/unet')`, which contained in some files will result in a fully reusable project file.
        Source Path: {project_path}  

        env:       
        unet7                 *  /media/zhangbolin/conda_envs/unet7
        """
        write_txt(content=note, file_path=os.path.join(output_dir, "note.txt"))

        
