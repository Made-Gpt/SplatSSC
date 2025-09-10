import torch
import numpy as np

from mmengine.registry import Registry
OPENOCC_DATASET = Registry('openocc_dataset')
OPENOCC_DATAWRAPPER = Registry('openocc_datawrapper')
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from .dataset_wrapper_scannet_occ import Scannet_Scene_Occ_DatasetWrapper
from .dataset_scannet_occ_openocc import Scannet_Scene_OpenOccupancy_Dataset


def custom_collate_fn_default(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, np.ndarray):
            data_tuple.append(torch.from_numpy(np.stack([d[i] for d in data])))
        elif isinstance(item, (dict, str)):
            data_tuple.append([d[i] for d in data])
        elif item is None:
            data_tuple.append(None)
        else:
            raise NotImplementedError
    return data_tuple


def custom_collate_fn_global(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, np.ndarray):
            data_tuple.append(torch.from_numpy(np.stack([d[i] for d in data])))
        elif isinstance(item, (dict, str)):
            data_tuple.append([d[i] for d in data])
        elif item is None:
            data_tuple.append(None)
        else:
            raise NotImplementedError
    return data_tuple


# These lists should be defined globally or passed as arguments if they change
METAS_TENSOR_KEYS_INV = [
    'depth_gt_np_valid', 'depth_gt_np', 'name', 'cam2img', 'world2img',
    'rgb_path', 'depth_path', 'num_depth', 'occ_mask_valid', 'occ_mask_valid_fov',
    'img_shape', 'img_aug_matrix', 'cam_vox_range', 'pix_z'
]
METAS_SQUEEZE = ['depth_gt_np', 'depth_gt_np_valid', 'img_depthbranch', 'rgb', 'label']

# TRULY not tensors (like filenames)
NON_TENSOR_KEYS = ['name', 'rgb_path', 'depth_path']


def custom_collate_fn_mono(batch):
    """
    The definitive, robust collate function that respects the user's vital logic.
    """
    imgs = [item[0] for item in batch]
    metas_list = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # --- Step 1: Stack images and labels ---
    imgs_batch = torch.from_numpy(np.stack(imgs, axis=0))
    labels_batch = torch.from_numpy(np.stack(labels, axis=0))

    # --- Step 2: Process metadata dictionaries with robust logic ---
    meta_batch_dict = {}
    all_keys = set().union(*[d.keys() for d in metas_list])

    for k in all_keys:
        v_list = [d.get(k) for d in metas_list if k in d]

        # First, handle keys that should always remain as lists (e.g., filenames)
        if k in NON_TENSOR_KEYS:
            meta_batch_dict[k] = v_list
            continue

        # For all other keys, attempt to stack them into a tensor.
        try:
            # Proactively convert elements to tensors. This handles many simple cases.
            tensors_to_stack = [torch.as_tensor(v) for v in v_list]
            stacked_tensor = torch.stack(tensors_to_stack, dim=0)

            if k in METAS_SQUEEZE:
                stacked_tensor = stacked_tensor.squeeze(1)

            meta_batch_dict[k] = stacked_tensor
        except Exception as e:
            # --- THIS IS THE VITAL FALLBACK LOGIC YOU PROVIDED ---
            # If the general stacking fails, apply the special handling.
            # This is crucial for keys like 'cam2img' which might be numpy arrays.
            if isinstance(v_list[0], np.ndarray):
                try:
                    # Use numpy's stack, which can be more forgiving
                    stacked_numpy = np.stack(v_list, axis=0)
                    meta_batch_dict[k] = torch.from_numpy(stacked_numpy)
                except ValueError:
                    # This can happen if numpy arrays have different shapes
                    print(
                        f"--> [DataLoader Warning] Could not stack numpy array for key '{k}' due to shape mismatch. Keeping as list.")
                    meta_batch_dict[k] = v_list
            else:
                # If it's not a numpy array and still fails, it's likely unstackable.
                print(f"--> [DataLoader Warning] Could not stack key '{k}'. Error: {e}. Keeping as list.")
                meta_batch_dict[k] = v_list

    return imgs_batch, meta_batch_dict, labels_batch


def build_dataloader(
            train_dataset_config,
            val_dataset_config,
            train_wrapper_config,
            val_wrapper_config,
            train_loader_config,
            val_loader_config,
            data_type="mono",
            dist=False,
    ):
    train_dataset = OPENOCC_DATASET.build(train_dataset_config)
    val_dataset = OPENOCC_DATASET.build(val_dataset_config)

    train_wrapper = OPENOCC_DATAWRAPPER.build(train_wrapper_config, default_args={'in_dataset': train_dataset})
    val_wrapper = OPENOCC_DATAWRAPPER.build(val_wrapper_config, default_args={'in_dataset': val_dataset})

    train_sampler = val_sampler = None
    if dist:
        train_sampler = DistributedSampler(train_wrapper, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_wrapper, shuffle=False, drop_last=False)

    if data_type == 'mono':
        custom_collate_fn = custom_collate_fn_mono
    elif data_type == 'embodied':
        custom_collate_fn = custom_collate_fn_global
    else:
        custom_collate_fn = custom_collate_fn_default
        # raise ValueError(f"illegal data type {data_type}, use 'mono' or 'embodied'")

    train_dataset_loader = DataLoader(dataset=train_wrapper,
                                    batch_size=train_loader_config["batch_size"],
                                    collate_fn=custom_collate_fn,
                                    shuffle=False if dist else train_loader_config["shuffle"],
                                    sampler=train_sampler,
                                    num_workers=train_loader_config["num_workers"],
                                    pin_memory=True,
                                    # persistent_workers=True
                                    )
    val_dataset_loader = DataLoader(dataset=val_wrapper,
                                    batch_size=val_loader_config["batch_size"],
                                    collate_fn=custom_collate_fn,
                                    shuffle=False if dist else val_loader_config["shuffle"],
                                    sampler=val_sampler,
                                    num_workers=val_loader_config["num_workers"],
                                    pin_memory=True,
                                    # persistent_workers=True
                                    )

    return train_dataset_loader, val_dataset_loader