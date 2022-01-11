from haven import haven_utils as hu

EXP_GROUPS = {

    "pretrain":  {
        "agent":'ddpg',
        "reward_free": false,
        # task settings
        "task": 'walker_stand',
        "obs_type": 'states', # [states, pixels]
        "frame_stack": 3, # only works if obs_type=pixels
        "action_repeat": 1, # set to 2 for pixels
        "discount": 0.99,
        # train settings
        "num_train_frames": 2000010,
        "num_seed_frames": 4000,
        # eval
        "eval_every_frames": 10000,
        "num_eval_episodes": 10,
        # pretrained
        "snapshot_ts": 100000,
        "snapshot_base_dir": './pretrained_models',
        # replay buffer
        "replay_buffer_size": 1000000,
        "replay_buffer_num_workers": 4,
        "batch_size": 1024,
        "nstep": 3,
        "update_encoder": false, # can be either true or false depending if we want to fine-tune encoder
        # misc
        "seed": 1,
        "device": cuda,
        "save_video": true,
        "save_train_video": false,
        "use_tb": false,
        "use_wandb": false,
        # experiment
        "experiment": exp,
                #"dataset": "imagenet" # the trainval.py will look into os.path.join(args.dataroot, dataset)
    },
    }
EXP_GROUPS = {k:hu.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}