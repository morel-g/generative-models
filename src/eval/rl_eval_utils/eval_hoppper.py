# Original source for hopper visualization: https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/reinforcement_learning_with_diffusers.ipynb#scrollTo=Yg9JiztlpH1o
import os
import warnings
import numpy as np
from PIL import Image, ImageDraw
from src.eval.plot_utils import get_font
from src.utils import ensure_directory_exists

import traceback

# missing_packages = []

# try:
#     import gym
# except ImportError:
#     missing_packages.append("gym")

# try:
#     import mujoco_py as mjc
# except ImportError:
#     missing_packages.append("mujoco_py")

# if missing_packages:
#     warning_message = "Warning: The following packages could not be imported: {}. RL features may not work.".format(
#         ", ".join(missing_packages)
#     )
#     warnings.warn(warning_message, ImportWarning)


def save_hopper_with_render(
    renderer,
    observations,
    output_dir,
    name="rl_samples.gif",
    titles=None,
    speed_factor=3,
):
    images = []
    for rollout in observations:
        # [horizon x height x width x channels]
        img = renderer._renders(rollout, partial=True)
        images.append(img)

    # [horizon x height x (batch_size * width) x channels]
    concatenated_images = np.concatenate(images, axis=2)

    # Reshape the images for saving as GIF
    reshaped_images = [
        concatenated_images[i] for i in range(concatenated_images.shape[0])
    ]

    pil_images = []
    for image in reshaped_images:
        pil_img = Image.fromarray(image)
        pil_img.info["duration"] = 10 * speed_factor

        if titles is not None:
            animation_width = reshaped_images[0].shape[1] // len(titles)
            add_gif_titles(pil_img, titles, animation_width)

        pil_images.append(pil_img)

    # Save as GIF
    pil_images[0].save(
        os.path.join(output_dir, name),
        save_all=True,
        append_images=pil_images[1:],
        loop=0,
        duration=pil_images[0].info["duration"],
    )


def add_gif_titles(img, titles, animation_width):
    draw = ImageDraw.Draw(img)
    font = get_font(size=14)
    for i, title in enumerate(titles):
        # Calculate position for each title
        title_position = (
            i * animation_width + animation_width / 2.0 - 10,
            10,
        )  # Adjust position as needed
        draw.text(title_position, title, font=font, fill=(255, 255, 255))


def env_map(env_name):
    """
    map D4RL dataset names to custom fully-observed
    variants for rendering
    """
    if "halfcheetah" in env_name:
        return "HalfCheetahFullObs-v2"
    elif "hopper" in env_name:
        return "HopperFullObs-v2"
    elif "walker2d" in env_name:
        return "Walker2dFullObs-v2"
    else:
        return env_name


def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask


def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x


def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f"[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, "
            f"but got state of size {state.size}"
        )
        state = state[: qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])


class MuJoCoRenderer:
    """
    default mujoco renderer
    """

    def __init__(self, env):
        import mujoco_py as mjc
        import gym

        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        ## @TODO : clean up
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)  # , device_id=0)
        # except:
        except Exception as e:
            print(
                "[ utils/rendering ] Warning: could not initialize offscreen renderer"
            )
            print("Error details:")
            traceback.print_exc()
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate(
            [
                np.zeros(1),
                observation,
            ]
        )
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate(
            [
                xpos[:, None],
                observations,
            ],
            axis=-1,
        )
        return states

    def render(
        self,
        observation,
        dim=256,
        partial=False,
        qvel=True,
        render_kwargs=None,
        conditions=None,
    ):
        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                "trackbodyid": 2,
                "distance": 3,
                "lookat": [xpos, -0.5, 1],
                "elevation": -20,
            }

        for key, val in render_kwargs.items():
            if key == "lookat":
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

    def save_state_samples(
        self, state, output_dir, name="hopper_samples", titles=None, speed_factor=3
    ):
        if not name.endswith(".gif"):
            name += ".gif"
        save_hopper_with_render(self, state, output_dir, name, titles, speed_factor)

    def save_state_trajectories(
        self, state_traj, output_dir, name="hopper", titles=None, speed_factor=3
    ):
        if name.endswith(".gif"):
            name = name[:-4]
        # Set batch_dim first
        new_dims = (1, 0) + tuple(range(2, state_traj.ndim))
        state_traj = state_traj.transpose(*new_dims)

        for i, xi in enumerate(state_traj):
            save_hopper_with_render(
                self, xi, output_dir, name + f"_traj_{i}.gif", titles, speed_factor
            )
