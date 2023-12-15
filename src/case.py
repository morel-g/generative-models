class Case:
    """
    The Case class is a collection of constant string values that represent
    various parameters and configurations.
    These constants can be used throughout the application to maintain
    consistency and avoid hardcoding string values.
    """

    # Problem type 2D: These constants define different types of 2D datasets
    # or problems.
    two_spirals = "two_spirals"
    moons = "moons"
    gaussian = "gaussian"
    cross_gaussians = "cross gaussians"
    swissroll = "swissroll"
    joint_gaussian = "joint_gaussian"
    eight_gaussians = "eight_gaussians"
    conditionnal8gaussians = "conditionnal8gaussians"
    pinwheel = "pinwheel"
    checkerboard = "checkerboard"
    uniform = "uniform"
    circles = "circles"
    multimodal_swissroll = "multimodal_swissroll"
    # Discrete toy datasets
    two_spirals_discrete = "two_spirals_discrete"
    moons_discrete = "moons_discrete"
    gaussian_discrete = "gaussian_discrete"
    cross_gaussians_discrete = "cross_gaussians_discrete"
    swissroll_discrete = "swissroll_discrete"
    joint_gaussian_discrete = "joint_gaussian_discrete"
    eight_gaussians_discrete = "eight_gaussians_discrete"
    conditionnal8gaussians_discrete = "conditionnal8gaussians_discrete"
    pinwheel_discrete = "pinwheel_discrete"
    checkerboard_discrete = "checkerboard_discrete"
    uniform_discrete = "uniform_discrete"
    circles_discrete = "circles_discrete"
    multimodal_swissroll_discrete = "multimodal_swissroll_discrete"

    # Other dataset: These constants represent different datasets commonly
    # used in the machine learning domain.
    mnist = "mnist"
    fashion_mnist = "fashion_mnist"
    cifar10 = "cifar10"
    cifar10_grayscale = "cifar10_grayscale"
    audio_diffusion_256 = "audio_diffusion_256"
    audio_diffusion_64 = "audio_diffusion_64"
    wiki = "wiki"
    lm1b_short = "lm1b_short"
    lm1b = "lm1b"
    hopper_medium_v2 = "hopper-medium-v2"
    maze2d_umaze_v1 = "maze2d-umaze-v1"
    maze2d_medium_v1 = "maze2d-medium-v1"
    maze2d_large_v1 = "maze2d-large-v1"
    custom_data = "custom_data"

    # Tokenizer
    gpt2 = "gpt2"

    # Transition matrix
    uniform = "uniform"
    absorbing = "absorbing"

    # Architecture type: Constants representing different neural network
    # architectures.
    ncsnpp = "ncsnpp"
    u_net = "u_net"
    u_net_1d = "u_net_1d"
    vector_field = "vector_field"
    transformer = "transformer"

    # Functions: Activation or mathematical functions that can be utilized
    # in models.
    tanh = "tanh"
    log_cosh = "log_cosh"
    relu = "relu"
    silu = "silu"

    # Model type: Different modeling techniques or strategies.
    stochastic_interpolant = "stochastic_interpolant"
    denoise_model = "denoise_model"
    score_model = "score_model"
    score_model_critical_damped = "score_model_critical_damped"
    d3pm = "d3pm"

    # Type of the interpolant for the stochastic interpolant model
    linear = "linear"
    linear_scale = "linear_scale"
    poly = "poly"
    bgk = "bgk"
    trigonometric = "trigonometric"

    # Discretization scheme: Constants representing different numerical
    # schemes for discretizing mathematical models.
    euler_implicit = "euler_implicit"
    euler_explicit = "euler_explicit"
    diffusion = "diffusion"
    anderson = "anderson"

    # Various methods or techniques used in the application.
    classic_score = "classic_score"
    time_exp = "time_exp"
    sliced_score = "sliced_score"

    # PDE case: Constants related to different types of partial differential
    # equations.
    constant = "constant"
    vanilla = "vanilla"

    # Decay case: Constants representing different types of decay methods or
    # strategies.
    no_decay = "no_decay"
    exp = "exp"
    vanilla_sigma = "vanilla_sigma"

    # Beta case: beta choive when considering a score base model.
    constant = "constant"
    vanilla = "vanilla"

    # Scheduler: Constants representing different learning rate scheduling
    # strategies.
    step_lr = "step_lr"
    cosine_with_warmup = "cosine_with_warmup"

    # Conditioning case
    conditioning_rl_first_last = "conditioning_rl_first_last"

    # FID version: Constants related to different versions of
    # FID (Frechet Inception Distance) metric.
    fid_v1 = "fid_v1"
    fid_v3 = "fid_v3"
    fid_metrics_v3 = "fid_metrics_v3"

    # Embeding type for the u_net architecture
    no_embeding = "no_embeding"
    fourier = "fourier"
    positional = "positional"

    # Noise addition for the stochastic interpolant model
    linear_noise = "linear_noise"
    sqrt_noise = "sqrt_noise"
    # EMA interval
    step = "step"
    epoch = "epoch"
