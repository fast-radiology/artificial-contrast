import setuptools

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements


install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]

setuptools.setup(
    name="artificial_contrast",
    setup_cfg=True,
    install_requires=reqs,
    packages=["artificial_contrast"],
    package_dir={"artificial_contrast": "artificial_contrast"},
    python_requires='>=3.6',
)
