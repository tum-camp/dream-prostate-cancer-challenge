import os
import os.path
import sys

DISTNAME = 'scikit-survival'
DESCRIPTION = 'Survival Models for the Prostate Cancer DREAM Challenge'
MAINTAINER = 'Sebastian Pölsterl'
MAINTAINER_EMAIL = 'sebastian.poelsterl@tum.de'
URL = 'https://www.synapse.org/#!Synapse:syn3647478'


VERSION = "1.0.1"


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('survival')

    return config


def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license="GNU General Public License version 3",
                    url=URL,
                    version=VERSION,
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Programming Language :: Python :: 3.3',
                                 'Programming Language :: Python :: 3.4',
                                 ],
                    scripts=['scripts/model_1a_survival.py',
                             'scripts/validate-survival.py',
                             'scripts/model_1b_regression.py',
                             'scripts/validate-regression.py',
                             'scripts/model_2_classification.py',
                             'scripts/validate-classifier.py'],
                    install_requires=[
                        'ipython',
                        'numpy>=1.9.0',
                        'numexpr>=2.4',
                        'pandas==0.15.2',
                        'pymongo'
                        'rpy2',
                        'scikit-learn>=0.16.0',
                        'scipy',
                        'seaborn==0.5.1',
                        'six',
                    ],
                    )

    if (len(sys.argv) >= 2
        and ('--help' in sys.argv[1:] or sys.argv[1]
        in ('--help-commands', 'egg_info', '--version', 'clean'))):

        from setuptools import setup

        metadata['version'] = VERSION
    else:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    py_version = sys.version_info[:2]
    if py_version < (3, 3):
        raise RuntimeError('Python 3.3 or later is required')

    setup_package()
