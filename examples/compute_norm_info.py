from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.visualization.vis import plot_agent_batch

import numpy as np
import torch
from torch import Tensor

import os

def main(dataset_to_use, dataset_loader_to_use, hist_sec = 1.0, fut_sec = 2.0, steps = None, agent_types = [AgentType.VEHICLE]):
    dt = 0.1

    if "nusc" in dataset_to_use:
        interaction_d = 30
    elif "lyft" in dataset_to_use:
        interaction_d = 50
    elif "nuplan" in dataset_to_use:
        interaction_d = 30
    else:
        raise
        
    if dataset_loader_to_use == "unified":
        dataset = UnifiedDataset(
            desired_data=[dataset_to_use],
            centric="agent",
            desired_dt=dt,
            history_sec=(hist_sec, hist_sec),
            future_sec=(fut_sec, fut_sec), # This should be consistent with the horizon diffusion model uses
            only_types=agent_types,
            only_predict=agent_types,
            agent_interaction_distances=defaultdict(lambda: interaction_d),
            incl_robot_future=False,
            incl_raster_map=False,
            raster_map_params={"px_per_m": 2, "map_size_px": 224, "offset_frac_xy": (-0.5, 0.0)},
            # augmentations=[noise_hists],
            data_dirs={
                "nusc_trainval": "../behavior-generation-dataset/nuscenes",
                "nusc_mini": "../behavior-generation-dataset/nuscenes",
                "lyft_sample": "../behavior-generation-dataset/lyft_prediction/scenes/sample.zarr",
                "lyft_val": "../behavior-generation-dataset/lyft_prediction/scenes/validate.zarr",
                "lyft_train": "../behavior-generation-dataset/lyft_prediction/scenes/train.zarr",
                "nuplan_mini": "../behavior-generation-dataset/nuplan/dataset/nuplan-v1.1",
            },
            cache_location="~/.unified_data_cache",
            num_workers=os.cpu_count()//2,
            rebuild_cache=False,
            rebuild_maps=False,
            standardize_data=True,
            max_agent_num=20
        )
        print(f"# Data Samples: {len(dataset):,}")

        dataloader = DataLoader(
            dataset,
            batch_size=50,
            shuffle=True,
            collate_fn=dataset.get_collate_fn(),
            num_workers=6,
        )
    elif dataset_loader_to_use == 'l5kit':
        assert dataset_to_use in ["lyft_val", "lyft_train"]
        from tbsim.datasets.l5kit_datamodules import L5MixedDataModule
        from tbsim.configs.registry import get_registered_experiment_config
        from tbsim.utils.config_utils import get_experiment_config_from_file, translate_l5kit_cfg
        
        config_name = 'l5_bc'
        # config_name = None
        config_file = '../behavior-generation/diffuser_trained_models/test/run144_lyft/config.json'
        
        if config_name is not None:
            cfg = get_registered_experiment_config(config_name)
            cfg.train.dataset_path = '../behavior-generation-dataset/lyft_prediction'
        elif config_file is not None:
            # Update default config with external json file
            cfg = get_experiment_config_from_file(config_file, locked=False)


        cfg.lock()  # Make config read-only
        if not cfg.devices.num_gpus > 1:
            # Override strategy when training on a single GPU
            with cfg.train.unlocked():
                cfg.train.parallel_strategy = None

        l5_config = translate_l5kit_cfg(cfg)

        datamodule = L5MixedDataModule(l5_config=l5_config, train_config=cfg.train)
        datamodule.setup()
        if dataset_to_use == "lyft_val":
            dataloader = datamodule.val_dataloader()
        elif dataset_to_use == "lyft_train":
            dataloader = datamodule.train_dataloader()
    else:
        raise 


    batch: AgentBatch
    compile_data = {
        'ego_fut' : [],
        'ego_hist' : [],
        'neighbor_hist' : []
    }
    for i, batch in enumerate(tqdm(dataloader)):
        # print(batch.scene_ids)
        # print(batch.maps.size())
        # plot_agent_batch(batch, batch_idx=0, rgb_idx_groups=([1], [0], [1]))

        # normalize over future traj
        past_traj: Tensor = batch.agent_hist.cuda()
        future_traj: Tensor = batch.agent_fut.cuda()

        hist_pos, hist_yaw, hist_speed, _ = trajdata2posyawspeed(past_traj, nan_to_zero=False)
        curr_speed = hist_speed[..., -1]

        fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(future_traj, nan_to_zero=False)

        traj_state = torch.cat(
                (fut_pos, fut_yaw), dim=2)

        traj_state_and_action = convert_state_to_state_and_action(traj_state, curr_speed, dt).reshape((-1, 6))

        # B*T x 6 where (x, y, vel, yaw, acc, yawvel)
        # print(traj_state_and_action.size())
        compile_data['ego_fut'].append(traj_state_and_action.cpu().numpy())

        # ego history
        ego_lw = batch.agent_hist_extent[:,:,:2].cuda()
        ego_hist_state = torch.cat((hist_pos, hist_speed.unsqueeze(-1), ego_lw), dim=-1).reshape((-1, 5))
        compile_data['ego_hist'].append(ego_hist_state.cpu().numpy())

        # neighbor history
        neigh_hist_pos, _, neigh_hist_speed, neigh_mask = trajdata2posyawspeed(batch.neigh_hist.cuda(), nan_to_zero=False)
        neigh_lw = batch.neigh_hist_extents[...,:2].cuda()
        neigh_state = torch.cat((neigh_hist_pos, neigh_hist_speed.unsqueeze(-1), neigh_lw), dim=-1)
        # only want steps from neighbors that are valid
        neigh_state = neigh_state[neigh_mask]
        compile_data['neighbor_hist'].append(neigh_state.cpu().numpy())

        if steps is not None and i > steps:
            break

    
    compile_data = {state_name:np.concatenate(state_list, axis=0) for state_name, state_list in compile_data.items()}
    path = 'examples/traj_data_'+dataset_to_use
    np.savez(path, **compile_data)
    print('data saved at', path)

    # traj_state_and_action_list = []

    # batch: AgentBatch
    # cum_mean = None
    # cum_std = None
    # all_max = np.ones((6)) * -np.inf
    # all_min = np.ones((6)) * np.inf
    # loop_num = 1
    # for loop_idx in range(loop_num): # one for mean, one for std
    #     curr_sum = torch.zeros((6)).cuda()
    #     num_samps = torch.zeros((6)).cuda()


    #     for i, batch in enumerate(tqdm(dataloader)):
    #         # print(type(batch))
    #         # print(batch.keys()) 
    #         # print(batch.scene_ids)
    #         # print(batch.maps.size())
    #         # plot_agent_batch(batch, batch_idx=0, rgb_idx_groups=([1], [0], [1]))

    #         if dataset_loader_to_use == "unified":
    #             # normalize over future traj
    #             past_traj: Tensor = batch.agent_hist.cuda()
    #             future_traj: Tensor = batch.agent_fut.cuda()

    #             _, _, hist_speed, _ = trajdata2posyawspeed(past_traj, nan_to_zero=False)
    #             curr_speed = hist_speed[..., -1]

    #             fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(future_traj, nan_to_zero=False)

    #             traj_state = torch.cat(
    #                     (fut_pos, fut_yaw), dim=2)

                
    #         elif dataset_loader_to_use == 'l5':
    #             fut_pos = batch['target_positions'].cuda()
    #             fut_yaw = batch['target_yaws'].cuda()
    #             traj_state = torch.cat(
    #                 (fut_pos, fut_yaw), dim=2)

    #             curr_speed = batch['curr_speed'].cuda()
    #         else:
    #             raise

    #         traj_state_and_action = convert_state_to_state_and_action(traj_state, curr_speed, dt).reshape((-1, 6))
    #         if loop_idx == 0:
    #             traj_state_and_action_np = traj_state_and_action.cpu().detach().numpy()
    #             # print('traj_state_and_action_np.shape:', traj_state_and_action_np.shape)
    #             traj_state_and_action_list.append(traj_state_and_action_np)
            
    #         # B*T x 6 where (x, y, vel, yaw, acc, yawvel)
    #         # print(traj_state_and_action.size())
    #         num_samps += traj_state_and_action.size(0) - torch.sum(torch.isnan(traj_state_and_action), dim=0)
    #         if loop_idx == 0:
    #             # computing mean
    #             curr_sum += torch.nansum(traj_state_and_action, dim=0)
    #             # extrema
    #             curr_max = np.nanmax(traj_state_and_action.cpu().numpy(), axis=0)
    #             all_max = np.maximum(all_max, curr_max)
    #             curr_min = np.nanmin(traj_state_and_action.cpu().numpy(), axis=0)
    #             all_min = np.minimum(all_min, curr_min)
    #         elif loop_idx == 1:
    #             # computing std
    #             curr_sum += torch.nansum((traj_state_and_action - cum_mean)**2, dim=0)
    #         else:
    #             print("Shouldn't be here")
    #             exit()

    #         # print('(%.05f, %.05f, %.05f, %.05f, %.05f, %.05f)' % tuple(curr_sum.cpu().numpy().tolist()))
    #         # print(num_samps)
    #         # print(curr_sum)
    #         # print(all_max)
    #         # print(all_min)

    #         if i > 70000:
    #             break

    #     print('(x, y, vel, yaw, acc, yawvel)')
    #     if loop_idx == 0:
    #         cum_mean = curr_sum / num_samps
    #         print('mean')
    #         print('(%.05f, %.05f, %.05f, %.05f, %.05f, %.05f)' % tuple(cum_mean.cpu().numpy().tolist()))
    #         print('max')
    #         print('(%.05f, %.05f, %.05f, %.05f, %.05f, %.05f)' % tuple(all_max.tolist()))
    #         print('min')
    #         print('(%.05f, %.05f, %.05f, %.05f, %.05f, %.05f)' % tuple(all_min.tolist()))
    #         # save data
    #         traj_state_and_action_list = np.concatenate(traj_state_and_action_list, axis=0)
    #         np.save('examples/traj_state_and_action_list_'+dataset_to_use+'.npy', traj_state_and_action_list)
    #     elif loop_idx == 1:
    #         cum_std = torch.sqrt(curr_sum / num_samps)
    #         print('std')
    #         print('(%.05f, %.05f, %.05f, %.05f, %.05f, %.05f)' % tuple(cum_std.cpu().numpy().tolist()))
    #     else:
    #         print("Shouldn't be here")
    #         exit()

    # print('================================')
    # print('FINAL')
    # print('mean')
    # print('(%.05f, %.05f, %.05f, %.05f, %.05f, %.05f)' % tuple(cum_mean.cpu().numpy().tolist()))
    # print('max')
    # print('(%.05f, %.05f, %.05f, %.05f, %.05f, %.05f)' % tuple(all_max.tolist()))
    # print('min')
    # print('(%.05f, %.05f, %.05f, %.05f, %.05f, %.05f)' % tuple(all_min.tolist()))
    # if loop_num > 1:
    #     print('std')
    #     print('(%.05f, %.05f, %.05f, %.05f, %.05f, %.05f)' % tuple(cum_std.cpu().numpy().tolist()))

def trajdata2posyawspeed(state, nan_to_zero=True):
    """Converts trajdata's state format to pos, yaw, and speed. Set Nans to 0s"""
    
    if state.shape[-1] == 7:  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
        state = torch.cat((state[...,:6],torch.sin(state[...,6:7]),torch.cos(state[...,6:7])),-1)
    else:
        assert state.shape[-1] == 8
    pos = state[..., :2]
    yaw = torch.atan2(state[..., [-2]], state[..., [-1]])
    speed = torch.norm(state[..., 2:4], dim=-1)
    mask = torch.bitwise_not(torch.max(torch.isnan(state), dim=-1)[0])
    if nan_to_zero:
        pos[torch.bitwise_not(mask)] = 0.
        yaw[torch.bitwise_not(mask)] = 0.
        speed[torch.bitwise_not(mask)] = 0.
    return pos, yaw, speed, mask

#
# Copied from our diffuser implementation so it's consistent
#
def angle_diff(theta1, theta2):
    '''
    :param theta1: angle 1 (..., 1)
    :param theta2: angle 2 (..., 1)
    :return diff: smallest angle difference between angles (..., 1)
    '''
    period = 2*np.pi
    diff = (theta1 - theta2 + period / 2) % period - period / 2
    diff[diff > np.pi] = diff[diff > np.pi] - (2 * np.pi)
    return diff

# TODO NEED TO HANLE MISSING FRAMES
def convert_state_to_state_and_action(traj_state, vel_init, dt):
    '''
    Infer vel and action (acc, yawvel) from state (x, y, yaw).
    Input:
        traj_state: (batch_size, num_steps, 3)
        vel_init: (batch_size,)
        dt: float
    Output:
        traj_state_and_action: (batch_size, num_steps, 6)
    '''
    target_pos = traj_state[:, :, :2]
    traj_yaw = traj_state[:, :, 2:]
    
    b = target_pos.size()[0]
    device = target_pos.get_device()

    # pre-pad with zero pos
    pos_init = torch.zeros(b, 1, 2, device=device)
    pos = torch.cat((pos_init, target_pos), dim=1)
    
    # pre-pad with zero pos
    yaw_init = torch.zeros(b, 1, 1, device=device) # data_batch["yaw"][:, None, None]
    yaw = torch.cat((yaw_init, traj_yaw), dim=1)

    # estimate speed from position and orientation
    vel_init = vel_init[:, None, None]
    vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * torch.cos(
        yaw[..., 1:, :]
    ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * torch.sin(
        yaw[..., 1:, :]
    )
    vel = torch.cat((vel_init, vel), dim=1)
    
    # m/s^2
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt
    # rad/s
    yawdiff = angle_diff(yaw[..., 1:, :], yaw[..., :-1, :])
    yawvel = yawdiff / dt

    pos, yaw, vel = pos[..., 1:, :], yaw[..., 1:, :], vel[..., 1:, :]

    traj_state_and_action = torch.cat((pos, vel, yaw, acc, yawvel), dim=2)

    return traj_state_and_action

def compute_info(path, sample_coeff=0.25):
    compile_data_npz = np.load(path)

    val_labels = {
        'ego_fut' : [    'x',       ' y',       'vel',      'yaw',     'acc',    'yawvel' ],
        'ego_hist' : [    'x',        'y',       'vel',      'len',     'width'    ],
        'neighbor_hist' : [    'x',        'y',       'vel',      'len',     'width'    ]
    }
    for i, state_name in enumerate(compile_data_npz.files):
        print(state_name)
        all_states = compile_data_npz[state_name]
        all_states = all_states[:int(all_states.shape[0]*sample_coeff)]
        
        # all_states = np.concatenate(state_list, axis=0)
        print(all_states.shape)
        print(np.sum(np.isnan(all_states)))

        # import matplotlib
        # import matplotlib.pyplot as plt
        # for di, dname in enumerate(['x', 'y', 'vel', 'yaw', 'acc', 'yawvel']):
        #     fig = plt.figure()
        #     plt.hist((all_state_and_action[:,di] - np_mean[di]) / np_std[di], bins=100)
        #     plt.title(dname)
        #     plt.show()
        #     plt.close(fig)

        # remove outliers before computing final statistics
        print('Removing outliers...')
        med = np.median(all_states, axis=0, keepdims=True)
        d = np.abs(all_states - med)
        mdev = np.std(all_states, axis=0, keepdims=True)
        s = d / mdev
        dev_thresh = 4.0
        chosen = s > dev_thresh
        all_states[chosen] = np.nan # reject outide of N deviations from median
        print('after outlier removal:')
        print(np.sum(chosen))
        print(np.sum(chosen, axis=0))
        print(np.sum(chosen) / (s.shape[0]*s.shape[1])) # removal rate

        out_mean = np.nanmean(all_states, axis=0)
        out_std = np.nanstd(all_states, axis=0)
        out_max = np.nanmax(all_states, axis=0)
        out_min = np.nanmin(all_states, axis=0)

        print('    '.join(val_labels[state_name]))
        out_fmt = ['( '] + ['%05f, ' for _ in val_labels[state_name]] + [' )']
        out_fmt = ''.join(out_fmt)
        print('out-mean')
        print(out_fmt % tuple(out_mean.tolist()))
        print('out-std')
        print(out_fmt % tuple(out_std.tolist()))
        print('out-max')
        print(out_fmt % tuple(out_max.tolist()))
        print('out-min')
        print(out_fmt % tuple(out_min.tolist()))

# def compute_info(path):
#     traj_state_and_action_list = np.load(path)
#     print('mean', np.nanmean(traj_state_and_action_list, axis=0))
#     print('std', np.nanstd(traj_state_and_action_list, axis=0))
#     print('max', np.nanmax(traj_state_and_action_list, axis=0))
#     print('min', np.nanmin(traj_state_and_action_list, axis=0))

if __name__ == "__main__":
    # 'nusc_trainval', 'lyft_train', 'lyft_sample', 'nuplan_mini'
    dataset_to_use = 'nuplan_mini' # 'nusc_trainval' # 'lyft_train'
    # 'unified', 'l5kit'
    dataset_loader_to_use = 'unified'
    hist_sec = 3.0 # 1.0
    fut_sec = 5.2 # 2.0
    steps = 50000
    agent_types = [AgentType.PEDESTRIAN] # [AgentType.VEHICLE]
    # main(dataset_to_use, dataset_loader_to_use, hist_sec, fut_sec, steps=steps, agent_types=agent_types)
    
    path = 'examples/traj_data_nuplan_mini.npz'
    compute_info(path)