from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="fastshap",
    author="Samuel Wilson",
    license="MIT",
    author_email="samwilson303@gmail.com",
    test_suite="tests",
    description="Fast SHAP kernel explainer",
    keywords=['shap', 'model explainability'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy','pandas', 'scikit-learn'],
    url="https://github.com/AnotherSamWilson/fastshap",
    packages=find_packages(exclude=["tests.*", "tests"]),
    classifiers=[
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
