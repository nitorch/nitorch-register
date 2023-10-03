from nitorch_cli.parser import ParseError
from .parser import parser, help
from ..register import entrypoint as entrypoint_register
import sys


def entrypoint(args=None):
    f"""Command-line interface for `simplereg`

    {help}

    """
    # Exceptions are dealt with here
    try:
        _entrypoint(args)
    except ParseError as e:
        print(help)
        print(f'[ERROR] {str(e)}', file=sys.stderr)
    except Exception as e:
        print(f'[ERROR] {str(e)}', file=sys.stderr)


def _entrypoint(args):
    """Command-line interface for `simplereg` without exception handling"""
    args = args or sys.argv[1:]

    options = parser.parse(args)
    if not options:
        return
    if options.help:
        print(help)
        return

    if options.verbose > 3:
        print(options)
        print('')

    args = ['--verbose', str(options.verbose)]
    args += [f'--{options.device[0]}']
    if options.device[1] is not None:
        args += [str(options.device[1])]
    args += ['@loss', options.loss]
    if options.loss in ('lcc', 'lgmm'):
        args += ['-p'] + [str(p) for p in options.kernel]
    if options.symmetric:
        args += ['--symmetric']
    args += ['@@fix', options.fixed]
    args += ['@@mov', options.moving]
    if options.init:
        args += ['-a', options.init]
    args += ['-o', str(options.moved)]
    args += ['-r', str(options.resliced)]
    args += ['@affine']
    if options.affine:
        args += ['affine']
    if not options.optimizer:
        if options.loss == 'nmi':
            options.optimizer = 'powell'
        else:
            options.optimizer = 'gn'
    if options.output:
        args += ['-o', options.output]
    args += ['@@optim', options.optimizer,
             '-n', options.max_iter,
             '-t', options.tolerance,
             '-s', options.search]
    args += ['@pyramid', '--levels']
    for level in options.pyramid:
        if isinstance(level, range):
            args += [str(l) for l in level]
        else:
            args += [str(level)]

    return entrypoint_register(args)
