import logging
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import AutoLocator

from utils.save_load import setup_visualization_dir
from visualization.color_generator import color_generator
from visualization.utils import keypoint_connections, jta_cam_int, rotation_3D, axes_order_3D

# from pygifsicle import optimize

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, dataset_name, parent_dir, images_dir):
        self.parent_dir = parent_dir
        self.images_dir = images_dir
        self.dataset_name = dataset_name

    def visualizer_3D(self, names, poses, cam_ext, cam_int, images_paths, observed_noise, gif_name, fig_size=(16, 12)):
        """
            visualizer_3D(poses, images_paths, fig_size) -> None
                @brief Draws a 3D figure with matplotlib (it can have multiple sub figures).
                    The function cv::This function draw multiple 3D poses alongside each other to make comparisons.
                :param names: name of each subplot. should be a list of strings.
                :param poses: torch.Tensor: should be a list of tensors of multiple outputs that you want to compare. (should have 4 dimensions)
                    shape of poses is like: [nun_comparisons, num_frames (which we create gif upon it), num_persons(in each frame), num_keypoints * dim]
                    Ex: poses.shape = [3, 16, 5, 51] means you want to compare 3 different groups of outputs each contain
                    5 persons with 17 joints (17 * 3).
                :param cam_ext: torch.Tensor or list of torch.Tensors: camera extrinsic parameters
                    shape of cam_ext is like: [num_comparisons, num_frames, 3, 4] which last two dimensions demonstrates (3, 4) matrix based on formal definitions
                    Ex: cam_ext.shape = [3, 16, 3, 4] means you want to compare 3 different groups of poses each contain
                    16 frame and unspecified number of persons. (for each frame basically we have a (3,4) matrix)
                :param cam_int: torch.Tensor or list of torch.Tensors: camera intrinsic parameters
                    shape of cam_int is like: [num_comparisons, 3, 3] which last two dimensions demonstrates (3, 3) matrix based on formal definitions
                    Ex: cam_int.shape = [3, 3] means you want to compare 3 different groups of poses each contain
                    (3, 3) matrix which demonstrate camera intrinsic parameters
                :param images_paths: list of tensors or list of numpy arrays: paths to specified outputs (scenes).
                    shape of images_paths is like: [num_comparisons, num_frames]
                    Ex: images_paths.shape = [3, 16] = means you want to compare 3 different groups of poses each have
                    16 images in it.
                :param observed_noise: torch.Tensor or list of torch.Tensors
                    shape of observed_noise is like: [nun_comparisons, num_frames (which we create gif upon it), num_persons(in each frame), num_keypoints]
                    Ex: masks.shape = [3, 16, 5, 17] just like 'poses'. The only difference here: we have 1 noise for each joint
                :param fig_size: tuple(size=2): size of matplotlib figure.
                    Ex: (8, 6)
                :param gif_name: str: name of generated output .gif file
                :return: None: generate a .gif file
        """
        poses = self.__clean_data(poses)
        if cam_ext and cam_int is not None:
            cam_ext = self.__clean_data(cam_ext)
            if images_paths:
                images_paths = self.__generate_images_path(images_paths)
            new_pose = []
            for i, group_pose in enumerate(poses):
                new_group_pose = []
                for j in range(len(group_pose)):
                    new_group_pose.append(
                        self.__scene_to_image(group_pose[j].unsqueeze(0), cam_ext[i], cam_int).tolist())
                new_pose.append(torch.tensor(new_group_pose).squeeze(1))
            self.visualizer_2D(names=names, poses=new_pose, images_paths=images_paths,
                               observed_noise=observed_noise, fig_size=fig_size,
                               gif_name=gif_name + '_2D_overlay')
        if self.dataset_name == 'jta':
            new_pose = []
            for i, group_pose in enumerate(poses):
                new_group_pose = []
                for j in range(len(group_pose)):
                    new_group_pose.append(self.__generate_JTA_2D_pose(group_pose[j].unsqueeze(0)).tolist())
                new_pose.append(torch.tensor(new_group_pose).squeeze(1))
            self.visualizer_2D(names=names, poses=new_pose, images_paths=images_paths, fig_size=fig_size,
                               observed_noise=observed_noise, gif_name=gif_name + "_2D_overlay")
        logger.info("start 3D visualizing.")
        if observed_noise is None or observed_noise == []:
            observed_noise = []
        else:
            observed_noise = self.__clean_data(observed_noise)
        max_axes = []
        min_axes = []
        for i in range(3):
            max_axes.append(int(max(map(lambda sub_fig_pose: torch.max(sub_fig_pose[:, :, i::3]) + 1, poses))))
            min_axes.append(int(min(map(lambda sub_fig_pose: torch.min(sub_fig_pose[:, :, i::3]) - 1, poses))))
        comparison_number = len(poses)
        axarr = []
        filenames = []
        save_dir = setup_visualization_dir(self.parent_dir)
        for j in range(len(poses[0])):
            fig = plt.figure(figsize=fig_size, dpi=100)
            axarr.append([])
            for i in range(len(poses)):
                noise = observed_noise if len(observed_noise) == len(poses[i]) else []
                axarr[j].append(fig.add_subplot(1, comparison_number, i + 1, projection='3d'))
                self.__create_plot(axarr[j][i], max_axes=max_axes, min_axes=min_axes)
                self.__generate_3D_figure(
                    i, all_poses=poses[i][j],
                    all_noises=noise[j] if j < len(noise) else None,
                    ax=axarr[j][i]
                )
                for _ in range(2):
                    filenames.append(os.path.join(save_dir, f'{j}.png'))
                if j == len(poses[0]) - 1:
                    for _ in range(3):
                        filenames.append(os.path.join(save_dir, f'{j}.png'))
                plt.title(names[i])
                plt.savefig(os.path.join(save_dir, f'{j}.png'), dpi=100)
            plt.close(fig)
        with imageio.get_writer(os.path.join(save_dir, f'{gif_name}.gif'), mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(filenames):
            os.remove(filename)
        # optimize(os.path.join(save_dir, f'{gif_name}.gif'))
        logger.info("end 3D visualizing.")

    def visualizer_2D(self, names, poses, images_paths, observed_noise, gif_name, fig_size=(24, 18)):
        """
             visualizer_2D(poses, images_paths, fig_size) -> gif
                @brief Draws a 2D figure with matplotlib (it can have multiple sub figures).
                    The function cv:: This function draw multiple 2D poses alongside each other to make comparisons
                    (in different outputs) you can draw many different persons together in one sub figure (scene) .
                :param names: name of each subplot. should be a list of strings.
                :param poses: torch.Tensor or list of torch.Tensors: 3D input pose
                    shape of poses is like: [nun_comparisons, num_frames (which we create gif upon it), num_persons(in each frame), num_keypoints * dim]
                    Ex: poses.shape = [3, 16, 5, 34] means you want to compare 3 different groups of outputs each contain
                    5 persons with 17 joints (17 * 2).
                :param images_paths: list or numpy.array: paths to specified outputs (scenes).
                    Ex: images_paths.shape = [3, 16]
                :param observed_noise: torch.Tensor or list of torch.Tensors
                    shape of observed_noise is like: [1, num_frames (which we create gif upon it), num_keypoints]
                    Ex: masks.shape = [16, 22] just like 'images_path'. The only difference here: we have 1 noise for each joint
                :param fig_size: tuple(size=2): size of matplotlib figure.
                    Ex: (8.6)
                :param gif_name: str: name of generated output .gif file
                :return None: generate a .gif file
        """
        logger.info("start 2D visualizing.")
        poses = self.__clean_data(poses)
        if observed_noise is None or observed_noise == []:
            observed_noise = []
        else:
            observed_noise = self.__clean_data(observed_noise)
        if images_paths is None or images_paths == []:
            images_paths = []
        else:
            images_paths = self.__generate_images_path(images_paths)
        subfig_size = len(poses)
        images = []
        for i, pose_group in enumerate(poses):
            images.append([])
            for j, pose in enumerate(pose_group):
                images[i].append(
                    self.__generate_2D_figure(
                        color_num=i, all_poses=pose,
                        all_noises=observed_noise[j] if j < len(observed_noise) else None,
                        image_path=images_paths[i][j] if i < len(images_paths) and j < len(images_paths[i]) else None
                    )
                )
        filenames = []
        save_dir = setup_visualization_dir(self.parent_dir)
        for plt_index in range(len(poses[0])):
            fig = plt.figure(figsize=np.array(fig_size), dpi=100)
            axarr = []
            for i in range(len(poses)):
                axarr.append(fig.add_subplot(1, subfig_size, i + 1))
                plt.title(names[i])
                axarr[i].imshow(images[i][plt_index])
            for _ in range(2):
                filenames.append(os.path.join(save_dir, f'{plt_index}.png'))
            if plt_index == len(poses[0]) - 1:
                for _ in range(3):
                    filenames.append(os.path.join(save_dir, f'{plt_index}.png'))
            plt.savefig(os.path.join(save_dir, f'{plt_index}.png'), dpi=100)
            plt.close(fig)
        with imageio.get_writer(os.path.join(save_dir, f'{gif_name}.gif'), mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)
        # optimize(os.path.join(save_dir, f'{gif_name}.gif'))
        logger.info("end 2D visualizing.")

    def __generate_3D_figure(self, color_num, all_poses, all_noises, ax):
        num_keypoints = all_poses.shape[-1] // 3
        poses = all_poses.reshape(all_poses.shape[0], num_keypoints, 3)
        if all_noises is None or all_noises == []:
            all_noises = torch.zeros(all_poses.shape[1] // 3)
        visualizing_keypoints = np.array(np.unique(keypoint_connections[self.dataset_name]))
        for i, keypoints in enumerate(poses):
            for ie, edge in enumerate(keypoint_connections[self.dataset_name]):
                ax.plot(xs=[keypoints[edge, 0][0], keypoints[edge, 0][1]],
                        zs=[keypoints[edge, 1][0], keypoints[edge, 1][1]],
                        ys=[keypoints[edge, 2][0], keypoints[edge, 2][1]], linewidth=2, label=r'$x=y=z$',
                        color=np.array(color_generator.get_color(color_num)) / 255)
            for k in visualizing_keypoints:
                ax.scatter(xs=keypoints[k, axes_order_3D[self.dataset_name][0]],
                           ys=keypoints[k, axes_order_3D[self.dataset_name][1]],
                           zs=keypoints[k, axes_order_3D[self.dataset_name][2]], s=2,
                           color=np.array([0, 255, 100]) / 255 if all_noises[k] == 0 else np.array(
                               [255, 40, 0]) / 255)

    def __generate_2D_figure(self, color_num, all_poses, all_noises=None, image_path=None):
        num_keypoints = all_poses.shape[-1] // 2
        poses = all_poses.reshape(all_poses.shape[0], num_keypoints, 2)
        if image_path is None:
            image = np.zeros((1080, 1920, 3)).astype(np.uint8)
        else:
            image = cv2.imread(image_path)
        if all_noises is None or all_noises == []:
            all_noises = torch.zeros(all_poses.shape[1] // 2)
        for i, keypoints in enumerate(poses):
            for keypoint in range(keypoints.shape[0]):
                for ie, edge in enumerate(keypoint_connections[self.dataset_name]):
                    if not ((keypoints[edge, 0][0] <= 0 or keypoints[edge, 1][0] <= 0) or (
                            keypoints[edge, 0][1] <= 0 or keypoints[edge, 1][1] <= 0)):
                        ''
                        cv2.line(image, (int(keypoints[edge, 0][0]), int(keypoints[edge, 1][0])),
                                 (int(keypoints[edge, 0][1]), int(keypoints[edge, 1][1])),
                                 color_generator.get_color(color_num), 4, lineType=cv2.LINE_AA)
            for keypoint in range(keypoints.shape[0]):
                cv2.circle(image, (int(keypoints[keypoint, 0]), int(keypoints[keypoint, 1])), 3,
                            (0, 255, 100) if all_noises[keypoint // 2] == 0 else (255, 50, 0), thickness=-1,
                            lineType=cv2.FILLED)
        return image

    def __create_plot(self, axe, max_axes, min_axes):
        axe.xaxis.set_major_locator(AutoLocator())
        axe.yaxis.set_major_locator(AutoLocator())
        axe.zaxis.set_major_locator(AutoLocator())
        range_axes = [(max_axes[i] - min_axes[i]) for i in range(len(max_axes))]
        true_range = [
            (min_axes[i] - (max(range_axes) - range_axes[i]) / 2, max_axes[i] + (max(range_axes) - range_axes[i]) / 2)
            for i in range(len(max_axes))]
        axe.set_aspect('auto')
        axe.view_init(elev=rotation_3D[self.dataset_name][0], azim=rotation_3D[self.dataset_name][1])
        axe.set_xlim(xmin=true_range[0][0],
                     xmax=true_range[0][1])
        axe.set_ylim(ymin=true_range[2][0],
                     ymax=true_range[2][1])
        axe.set_zlim(zmin=true_range[1][0],
                     zmax=true_range[1][1])

    @staticmethod
    def __scene_to_image(pose, cam_ext, cam_int):
        """
            scene_to_image(pose, cam_ext, cam_int) -> 2D_pose
                @brief this function project 3D locations with respect to camera into 2D pixels on body poses.
                :param pose: torch.Tensor: 3D input pose
                    shape of pose is like: [num_persons(in each frame), num_frames, num_keypoints * 3]
                    Ex: [2, 16, 72]
                :param cam_ext: torch.Tensor: camera extrinsic parameters
                    shape of cam_ext is like: [num_frames, 3, 4] which last two dimensions demonstrates (3, 4) matrix based on formal definitions
                    Ex: [16, 3, 4]
                :param cam_int: torch.Tensor: camera intrinsic parameters
                    shape of cam_int is like: [3, 3] which demonstrate (3, 3) matrix based on formal definitions
                :return 2d_pose: torch.Tensor: 2D projected pose
        """
        first_shape = pose.shape
        poses = pose.reshape(pose.shape[0], pose.shape[1], pose.shape[-1] // 3, 3)
        one_padding = torch.ones(poses.shape[0], poses.shape[1], pose.shape[-1] // 3, 1)

        poses = torch.cat((poses, one_padding), 3)
        poses = poses.transpose(1, 0)
        new_pose = []
        for frame_num, frame_data in enumerate(poses):
            for p_data in frame_data:
                new_data = []
                for joint_data in p_data:
                    new_joint_data = torch.matmul(cam_int, torch.matmul(cam_ext[frame_num][:3], joint_data))
                    new_data.append((new_joint_data[:2] / new_joint_data[-1]).tolist())
                new_pose.append(new_data)
        return torch.tensor(new_pose).reshape(first_shape[0], first_shape[1], 2 * first_shape[-1] // 3)

    @staticmethod
    def __generate_JTA_2D_pose(pose):
        first_shape = pose.shape
        poses = pose.reshape(pose.shape[0], pose.shape[1], pose.shape[-1] // 3, 3)
        new_pose = []
        for frame_num, frame_data in enumerate(poses):
            for p_data in frame_data:
                new_data = []
                for joint_data in p_data:
                    x_p = joint_data[0] / joint_data[2]
                    y_p = joint_data[1] / joint_data[2]
                    x = jta_cam_int[0][0] * x_p + jta_cam_int[0][2]
                    y = jta_cam_int[1][1] * y_p + jta_cam_int[1][2]
                    new_data.append([x, y])
                new_pose.append(new_data)
        return torch.tensor(new_pose).reshape(first_shape[0], first_shape[1], 2 * first_shape[-1] // 3)

    @staticmethod
    def __clean_data(input_data: list):
        new_data = []
        max_len = 0

        for i in range(len(input_data)):
            if input_data[i] is None:
                continue
            if len(input_data[i]) > max_len:
                max_len = len(input_data[i])
        for i, pose in enumerate(input_data):
            if input_data[i] is None:
                continue
            elif len(input_data[i]) < max_len:
                size = [1 for _ in range(len(pose.shape))]
                size[0] = max_len - len(input_data[i])
                last_row = pose[-1:]
                expended_data = last_row.repeat(size)
                expanded_data = torch.cat((pose, expended_data))
            else:
                expanded_data = pose
            new_data.append(expanded_data)
        return new_data

    def __generate_images_path(self, images_paths):
        if self.images_dir is None:
            return []
        new_images_path = []
        max_len = len(images_paths[0])
        for i in range(len(images_paths)):
            if len(images_paths[i]) > max_len:
                max_len = len(images_paths[i])
        for i, image_path in enumerate(images_paths):
            group_images_path = []
            for img in image_path:
                group_images_path.append(os.path.join(self.images_dir, img))
            if len(image_path) < max_len:
                last_path = image_path[-1]
                for i in range(max_len - len(image_path)):
                    group_images_path.append(os.path.join(self.images_dir, last_path))
            new_images_path.append(group_images_path)
        return new_images_path
