from src.case import Case

toy_continuous_data_type = [
    Case.two_spirals,
    Case.moons,
    Case.cross_gaussians,
    Case.swissroll,
    Case.joint_gaussian,
    Case.eight_gaussians,
    Case.pinwheel,
    Case.checkerboard,
    Case.uniform,
    Case.circles,
    Case.conditionnal8gaussians,
    Case.multimodal_swissroll,
]

toy_discrete_data_type = [
    Case.two_spirals_discrete,
    Case.moons_discrete,
    Case.cross_gaussians_discrete,
    Case.swissroll_discrete,
    Case.joint_gaussian_discrete,
    Case.eight_gaussians_discrete,
    Case.pinwheel_discrete,
    Case.checkerboard_discrete,
    Case.uniform_discrete,
    Case.circles_discrete,
    Case.conditionnal8gaussians_discrete,
    Case.multimodal_swissroll_discrete,
]

img_data_type = [
    Case.mnist,
    Case.fashion_mnist,
    Case.cifar10,
    Case.cifar10_grayscale,
    # Audio are also considered as images data type since the signal is
    # converted to a Mel spectogram
    Case.audio_diffusion_256,
    Case.audio_diffusion_64,
]

rl_data_type = [
    Case.hopper_medium_v2,
    Case.maze2d_umaze_v1,
    Case.maze2d_medium_v1,
    Case.maze2d_large_v1,
]

custom_data_type = [Case.custom_data]

text_data_type = [Case.wiki, Case.lm1b, Case.lm1b_short]

toy_data_type = toy_continuous_data_type + toy_discrete_data_type

audio_data_type = [Case.audio_diffusion_256, Case.audio_diffusion_64]

discrete_data_type = toy_discrete_data_type + text_data_type
