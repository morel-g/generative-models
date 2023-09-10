class Case:
    """
    The Case class is a collection of constant string values that represent
    various parameters and configurations.
    These constants can be used throughout the application to maintain
    consistency and avoid hardcoding string values.
    """

    # Problem type 2D: These constants define different types of 2D datasets
    # or problems.
    two_spirals = "2spirals"
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
    n_dim_gaussians = "n_dim_gaussians"
    multimodal_swissroll = "multimodal_swissroll"

    # Other dataset: These constants represent different datasets commonly
    # used in the machine learning domain.
    mnist = "mnist"
    fashion_mnist = "fashion_mnist"
    cifar10 = "cifar10"
    cifar10_grayscale = "cifar10_grayscale"

    # Architecture type: Constants representing different neural network
    # architectures.
    ncsnpp = "ncsnpp"
    u_net = "u_net"
    u_net_fashion_mnist = "u_net_fashion_mnist"
    vector_field = "vector_field"

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

    # FID version: Constants related to different versions of
    # FID (Frechet Inception Distance) metric.
    fid_v1 = "fid_v1"
    fid_v3 = "fid_v3"
    fid_metrics_v3 = "fid_metrics_v3"

    # Embeding type for the u_net architecture
    no_embeding = "no_embeding"
    fourier = "fourier"
