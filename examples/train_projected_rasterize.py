import os
import argparse
import numpy as np
import torch
import math
import json
import viser.transforms as vtf
import torchvision.transforms as transforms
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor, optim


# Add other necessary imports here
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for extracting points from an object file."
    )
    parser.add_argument(
        "object_file",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/in2/03790512/1a48f3b3c0ab0fdeba8e0a95fbfb7c9/models/model_normalized.obj",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/out2/params/out.txt",
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/out/img/1a48f3b3c0ab0fdeba8e0a95fbfb7c9/models/transforms.json",
    )
    parser.add_argument(
        "image",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/out/img/1a48f3b3c0ab0fdeba8e0a95fbfb7c9/models/000.png",
    )
    """
    parser.add_argument(
        "out_dir",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/out2/params",
    )
    parser.add_argument(
        "rotation_file",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/out/camera/1a48f3b3c0ab0fdeba8e0a95fbfb7c9/models/rotation.npy",
    )
    parser.add_argument(
        "elevation_file",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/out/camera/1a48f3b3c0ab0fdeba8e0a95fbfb7c9/models/elevation.npy",
    )
    """
    # Add other arguments as needed
    return parser.parse_args()


def extract_points_from_obj(input_file):
    points = []
    counter = 0
    try:
        with open(input_file, "r") as f:
            for line in f:
                if line.startswith("v "):
                    counter += 1
                    # Extracting vertices (points)
                    parts = line.split()
                    # Assuming format: v x y z
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])
    except Exception as e:
        print(f"Error extracting points from {input_file}: {e}")

    return points


def extract_transform_matrix(file_path, index):
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract the transform matrix at the specified index
    if 0 <= index < len(data["frames"]):
        transform_matrix = data["frames"][index]["transform_matrix"]
        return transform_matrix
    else:
        print(f"Index {index} is out of range.")
        return None


def show_image_from_tensor(tensor):
    # Convert the tensor to a NumPy array
    # print(tensor)
    image_array = tensor.cpu().numpy()

    # Reshape the array if necessary
    # if len(image_array.shape) == 4:
    #     image_array = np.squeeze(image_array, axis=0)

    # Plot the image
    plt.imshow(image_array)
    # print(image_array)
    plt.axis("on")  # Turn off axis
    plt.show()


def tensor_to_image(tensor, range_min=-1024, range_max=1024):
    # Convert the tensor to a NumPy array
    image_array = tensor.cpu().numpy()

    # Scale the values to fit within the desired range
    scaled_image_array = (
        (image_array - image_array.min()) / (image_array.max() - image_array.min())
    ) * (range_max - range_min) + range_min

    # Plot the image
    plt.imshow(
        scaled_image_array,
        cmap="gray",
        extent=[range_min, range_max, range_min, range_max],
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Image from Tensor")
    plt.colorbar()  # Add colorbar for reference
    plt.show()


def projection_matrix(znear, zfar, fovx, fovy, device="cpu"):
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    ).float()


def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return np.array(([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]]))


def image_path_to_tensor(image_path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


class TrainModel:
    def __init__(self, gt_image, args):
        self.device = torch.device("cuda:0")
        self.args = args
        self.gt_image = gt_image.to(device=self.device)
        self.mean = torch.tensor(
            extract_points_from_obj(args.object_file), device=self.device
        )
        self.num_of_points = self.mean.shape[0]
        self.H, self.W = self.gt_image[0], self.gt_image[1]

        self.init_gaussians()

    def init_gaussians(self):
        u = self.mean[:, 0]
        v = self.mean[:, 1]
        w = self.mean[:, 2]

        inner_values = [0.9, 0.7, 0.9]
        self.scales = torch.tensor(
            [inner_values] * self.mean.shape[0], device=self.device
        )
        self.glob_scale = 0.9
        quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.quats = torch.tensor(quats, device=self.device)
        self.rgbs = torch.rand(self.num_of_points, 3, device=self.device)
        self.opacities = torch.ones((self.mean.shape[0], 1), device=self.device)

        viewmat_np = np.array(extract_transform_matrix(self.args.json_file, 0))
        self.viewmat = torch.tensor(
            np.matmul(
                np.transpose(viewmat_np), rotate_x(-math.pi / 2, device=self.device)
            ),
            device=self.device,
        ).float()

        self.fx = 0.5 * float(self.H) / math.tan(0.5 * (math.pi / 2.0))
        self.fy = 0.5 * float(self.W) / math.tan(0.5 * (math.pi / 2.0))
        self.cx = self.H / 2  # cx  - cam.data.sensor_width = 32
        self.cy = self.W / 2

        self.block_width = 16
        self.clip_thresh = 0.01
        self.background = torch.zeros(3, device=self.device)

        self.means.requires_grad = False
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def train(self, iter, lr):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()

        for itr in range(iter):
            out_project = project_gaussians(
                self.means,
                self.scales,
                self.glob_scale,
                self.quats,
                self.viewmat,
                self.viewmat,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                self.H,
                self.W,
                self.block_width,
                self.clip_thresh,
            )

            xys, depth, radii, conics, compensation, num_tiles_hit, cov3d = out_project

            final_out = rasterize_gaussians(
                xys,
                depth,
                radii,
                conics,
                num_tiles_hit,
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities),
                self.H,
                self.W,
                self.block_width,
            )

            loss = mse_loss(final_out, self.gt_image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return final_out


def main():
    # Extract points from the object file
    device = torch.device("cuda:0")
    args = parse_arguments()
    img = image_path_to_tensor(args.image)

    trainer = TrainModel(img, args=args)

    final_out = trainer.train(1000, 0.01)

    # show_image_from_tensor(final_out)
    tensor_to_image(final_out)


if __name__ == "__main__":
    main()
