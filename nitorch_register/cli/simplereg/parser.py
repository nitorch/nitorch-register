from nitorch_cli.parser import CommandParser, Actions, Validations


help = r"""[nitorch] simplereg

Pairwise affine + nonlinear registration.
This command is merely a wrapper around nitorch register, that exposes 
fewer parameters and uses a more classical syntax.

note:
    Multiple pairs of images can be provided, using multiple -m/-f options.
    Similarly, multiple preprocessing, output, and losses options can be provided.
    If only one of these options is provided, it applies to all pairs of images.
    If multiple options are provided, they are applied in order (e.g., 
    the first occurrence of -l is applied to the first occurrence of -m/-f)
    Options that can appear multiple times are marked by [m].

usage: 
    nitorch simplereg -m MOVING -f FIX [options]
    
Input options:
[m] -m, --moving PATH               Moving image
[m] -f, --fixed PATH                Fixed image
[m] -i, --init PATH                 Initial affine to apply to `moving`.
[m]     --fwhm [0]                  Smoothing applied to the moving image.
[m]     --bound [dct2]              Boundary condition: {dct2,dct1,dst2,dst1,dft,zero,nearest}
[m]     --order [1]                 Interpolation order: {1..7}
[m]     --pad [0]                   Pad image before using it.
[m]     --rescale [[0], 95]         Rescale image so that its percentiles match (0, 1).
[m]     --discretize [256]          Discretize image into N bins (default: no)     
        
    

Output options:
    -o, --output PATH               Output path to transform [{dir}/{name}.lta]
[m] -x, --moved PATH                Output path to moved image [{dir}/{base}.moved{ext}]
[m] -r, --resliced PATH             Output path to resliced image [{dir}/{base}.resliced{ext}]
    
Loss options: multiple options (one per -p) can be provided
[m] -l, --loss                      Objective function. 
       [nmi]                            Normalized mutual information
        mse                             Mean squared error
        mad                             Median absolute deviation
        tuk                             Tukey's biweight function
        cc                              Correlation coefficient
        lcc                             Local correlation coefficient
        gmm                             Gaussian mixture likelihood
        lgmm                            Local Gaussian mixture likelihood
        emmi                            EM mutual information
[m] -k, --kernel [SIZE] [UNIT]      Kernel size (for lcc/lgmm). Unit can be {vox,mm,pct} [10 pct]
[m] -s, --symmetric                 Use a symmetric loss [false]
    
Optimization options:
    -a, --affine                    12-param affine [default is 6-param rigid]
    -p, --pyramid [LEVELS]          Pyramid levels to process [0:3]
    -z, --optimizer [OPT]           Optimizer {gn,powell,lbfgs} [auto]
    -n, --max-iter VAL              Max number of iterations [50]
    -t, --tolerance VAL             Tolerance for early stopping [1e-5]
        --line-search VAL           Number of line search iterations [wolfe]
    
Other options:
    -h, --help [LEVEL]              Display this help
    -v, --verbose [LVL]             Level of verbosity [1=print], 2=print more, 3=plot
    --cpu [THREADS], --gpu          Device to use [cpu]

"""


def number_or_str(type=float):
    def _number_or_str(x):
        try:
            return type(x)
        except ValueError:
            return x
    return _number_or_str


def bool_or_str(x):
    if x.lower() in ('true', 'yes'):
        return True
    if x.lower() in ('false', 'no'):
        return False
    try:
        return bool(int(x))
    except ValueError:
        return x


def parse_range(x):
    if ':' not in x:
        return int(x)
    x = x.split(':')
    if len(x) == 2:
        x = [*x, '']
    elif len(x) == 1:
        x = ['', *x, '']
    start, stop, step = x
    start = int(start or 0)
    step = int(step or 1)
    stop = int(stop)
    return range(start, stop, step)


def convert_device(device):
    def _convert(x):
        return (device, int(x))
    return _convert


parser = CommandParser('affreg', help=help)

# Input options:
#     -m, --moving PATH               Moving image
#     -f, --fixed PATH                Fixed image
#     -i, --init PATH                 Initial affine to apply to `moving`.
parser.add_option('moving', ('-m', '--moving'), nargs=1, help='Moving image')
parser.add_option('fixed', ('-f', '--fixed'), nargs=1, help='Fixed image')
parser.add_option('init', ('-i', '--init'), nargs=1, help='Initial affine')
# Output options:
#     -o, --output PATH               Output path to transform [{dir}/{name}.lta]
#     -x, --moved PATH                Output path to moved image [{dir}/{base}.moved{ext}]
#     -r, --resliced PATH             Output path to resliced image [{dir}/{base}.resliced{ext}]
parser.add_option('output', ('-o', '--output'), nargs=1, default='{dir}/{name}.lta', convert=bool_or_str)
parser.add_option('moved', ('-x', '--moved'), nargs=1, default='{dir}/{base}.moved{ext}', convert=bool_or_str)
parser.add_option('resliced', ('-r', '--resliced'), nargs=1, default=False, convert=bool_or_str)
# Loss options:
#     -l, --loss                      Objective function
#        [nmi]                            Normalized mutual information
#         mse                             Mean squared error
#         mad                             Median absolute deviation
#         tuk                             Tukey's biweight function
#         cc                              Correlation coefficient
#         lcc                             Local correlation coefficient
#         gmm                             Gaussian Mixture likelihood
#         lgmm                            Local Gaussian Mixture likelihood
#     -k, --kernel [SIZE] [UNIT]      Kernel size (for lcc/lgmm). Unit can be {vox,mm,pct} [10 pct]
#     -s, --symmetric                 Use a symmetric loss [false]
losses = ['nmi', 'mse', 'mad', 'tuk', 'cc', 'lcc', 'gmm', 'lgmm']
parser.add_option('loss', ('-l', '--loss'), nargs=1, default='nmi',
                  validation=Validations.choice(losses))
parser.add_option('kernel', ('-k', '--kernel'), nargs='+', default=[10, 'pct'],
                  convert=number_or_str(float))
parser.add_option('symmetric', ('-s', '--symmetric'), nargs='?', default=False,
                  convert=bool_or_str, action=Actions.store_true)
# Optimization options:
#     -a, --affine                    12-param affine [default is 6-param rigid]
#     -p, --pyramid [LEVELS]          Pyramid levels to process [0:3]
#     -z, --optimizer [OPT]           Optimizer {gn,powell,lbfgs} [auto]
#     -n, --max-iter VAL              Max number of iterations [50]
#     -t, --tolerance VAL             Tolerance for early stopping [1e-5]
#         --line-search VAL           Number of line search iterations [wolfe]
parser.add_option('affine', ('-a', '--affine'), nargs='?', default=False,
                  convert=bool_or_str, action=Actions.store_true)
parser.add_option('pyramid', ('-p', '--pyramid'), nargs='+', default=[range(3)],
                  convert=parse_range)
parser.add_option('pyramid', '--no-pyramid', nargs=0, action=Actions.store_false)
optims = ['gn', 'lbfgs', 'powell']
parser.add_option('optimizer', ('-z', '--optimizer'), nargs=1, validation=Validations.choice(optims))
parser.add_option('max_iter', ('-n', '--max-iter'), nargs=1, default=50, convert=int)
parser.add_option('tolerance', ('-t', '--tolerance'), nargs=1, default=1e-5, convert=float)
parser.add_option('search', '--line-search', nargs=1, default='wolfe', convert=number_or_str(int))
# Other options:
#     -h, --help [LEVEL]              Display this help
#     -v, --verbose [LVL]             Level of verbosity [1=print], 2=print more, 3=plot
#     --cpu [THREADS], --gpu          Device to use [cpu]
parser.add_option('verbose', ('-v', '--verbose'),
                  nargs='?', default=0, convert=int,
                  action=Actions.store_value(1),
                  help='Level of verbosity')
parser.add_option('device', '--gpu', default=('cpu', None),
                  nargs='?', convert=convert_device('gpu'),
                  action=Actions.store_value(('gpu', None)),
                  help='Use GPU (if available)')
parser.add_option('device', '--cpu',
                  nargs='?', convert=convert_device('cpu'),
                  action=Actions.store_value(('cpu', None)),
                  help='Use CPU')