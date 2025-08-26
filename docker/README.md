# Docker images for on premise testing

This directory contains Dockerfiles for building images for running on-prem tests that require internal Allen Institute resources. On-prem tests use GitHub self-hosted runners that will run tests on docker images built from these Dockerfiles.

Our light and on-prem tests are defined in [our workflow file](../.github/workflows/github-actions-ci.yml "Link to GitHub Actions workflow for light and on-prem tests").

- See [here](https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners) for more information on self-hosted runners.
- See [here](https://docs.github.com/en/actions/hosting-your-own-runners/adding-self-hosted-runners) for more information on adding self-hosted runners to a GitHub repository.

## Building images

If you are an Allen Institute developer, you will have instructions on how to access the machine running the
IPFX self-hosted runner.
